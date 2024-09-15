import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import tenseal as ts
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import random

# Carica il contesto TenSEAL
with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)

# Definizione del modello ottimale e semplice
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  

# Client FL con TenSEAL
class HomomorphicFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        

    def get_parameters(self, config):
        params = [param.cpu().detach().numpy() for param in self.net.parameters()]
        encrypted_params = [ts.ckks_vector(context, param.flatten()).serialize() for param in params]
        return encrypted_params

    def set_parameters(self, parameters):
        params = []
        for param in parameters:
            serialized_param = param.tobytes()
            ckks_vector = ts.lazy_ckks_vector_from(serialized_param)
            ckks_vector.link_context(context)
            decrypted_param = ckks_vector.decrypt()
            decrypted_param = np.array(decrypted_param)
            params.append(decrypted_param)

        params_dict = zip(self.net.state_dict().keys(), params)
        state_dict = {k: torch.Tensor(v.reshape(self.net.state_dict()[k].shape)) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=5)
        self.generate_confusion_matrix()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"partition_id": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, accuracy = test(self.net, self.valloader)
        return float(val_loss), len(self.valloader.dataset), {"val_loss": float(val_loss), "accuracy": float(accuracy)}

    def generate_confusion_matrix(self):
        """Genera e salva la matrice di confusione per questo client."""
        all_preds = []
        all_labels = []

        self.net.eval()
        with torch.no_grad():
            for images, labels in self.valloader:
                outputs = self.net(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["NORMAL", "PNEUMONIA"], 
                    yticklabels=["NORMAL", "PNEUMONIA"])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix Client {self.cid} (Absolute Counts)')
        plt.savefig(f'confusion_matrix_client_{self.cid}.png')
        plt.close()

# Funzione di training
def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Funzione di valutazione
def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return val_loss / len(testloader), accuracy

# Caricamento del dataset di Pneumonia
def load_balanced_datasets(partition_id, num_partitions=5):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = datasets.ImageFolder('DB/train', transform=transform)
    
    # Separare gli indici per classe
    normal_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
    pneumonia_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]
    
    # Assicurarsi che ogni partizione abbia un numero uguale di immagini per classe
    samples_per_class = min(len(normal_indices), len(pneumonia_indices)) // num_partitions
    
    start_idx = partition_id * samples_per_class
    end_idx = start_idx + samples_per_class
    
    partition_normal = normal_indices[start_idx:end_idx]
    partition_pneumonia = pneumonia_indices[start_idx:end_idx]
    
    # Combinare e mescolare gli indici
    partition_indices = partition_normal + partition_pneumonia
    random.shuffle(partition_indices)
    
    # Creare il subset per questa partizione
    partition_dataset = Subset(full_dataset, partition_indices)
    
    # Dividere in training e validation set (80% training, 20% validation)
    train_size = int(0.8 * len(partition_dataset))
    val_size = len(partition_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(partition_dataset, [train_size, val_size])
    
    # Ottenere le etichette per il training set
    train_targets = torch.tensor([full_dataset.targets[i] for i in train_dataset.indices])
    
    # Calcolare i pesi per il sampler
    class_sample_count = torch.bincount(train_targets)
    weight = 1. / class_sample_count.float()
    samples_weight = weight[train_targets]
    
    # Creare il WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader


# Funzione principale
def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--partition-id", type=int, required=True, help="ID of the partition to use")
    args = parser.parse_args()

    trainloader, valloader = load_balanced_datasets(partition_id=args.partition_id)

    net = PneumoniaCNN()

    fl.client.start_client(
        server_address="localhost:8080",
        client=HomomorphicFlowerClient(str(args.partition_id), net, trainloader, valloader).to_client()
    )

if __name__ == "__main__":
    main()
