import argparse
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flwr as fl
from opacus import PrivacyEngine
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import random

warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated")

parser = argparse.ArgumentParser(description='Federated Learning Client')
parser.add_argument('--partition-id', type=int, required=True, help='Partition ID')
args = parser.parse_args()

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

class FlowerClientWithDP(fl.client.NumPyClient):
    def __init__(self, model, partition_id, num_partitions=5, delta=1e-5):
        self.model = model
        self.partition_id = partition_id
        self.train_loader, self.val_loader = load_balanced_datasets(partition_id, num_partitions)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Inizializzazione del PrivacyEngine con delta
        
        self.privacy_engine = PrivacyEngine()

        # Collegamento del PrivacyEngine al modello e all'ottimizzatore
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=0.9,  # Controllo del rumore
            max_grad_norm=1.5,  # Clipping del gradiente
        )

        # Delta per la privacy differenziale
        self.delta = delta

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(5):
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        # Calcola l'epsilon dopo l'addestramento
        epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
        print(f"Training complete. (ε = {epsilon:.2f}, δ = {self.delta})")
        self.generate_confusion_matrix()

        loss_sum = 0
        with torch.no_grad():
            for images, labels in self.train_loader:
                outputs = self.model(images)
                loss_sum += self.criterion(outputs, labels).item() * len(images)
        average_loss = loss_sum / len(self.train_loader.dataset)
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": average_loss}
    

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * len(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / len(self.val_loader.dataset)
        return avg_loss, len(self.val_loader.dataset), {"loss": avg_loss, "accuracy": accuracy}
    
    def generate_confusion_matrix(self):
        """Genera e salva la matrice di confusione per le classi NORMAL e PNEUMONIA."""
        all_preds = []
        all_labels = []

        # Ottieni le predizioni e le etichette vere
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcola la confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Definisci le etichette per le classi
        class_names = ['NORMAL', 'PNEUMONIA']

        # Visualizza la confusion matrix senza normalizzazione
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix Client {self.partition_id} (Absolute Counts)')

        # Salva l'immagine con il nome del client
        plt.savefig(f'confusion_matrix_client_{self.partition_id}.png')
        plt.close()

def main():
    # Inizializza il modello
    model = PneumoniaCNN()
    
    # Crea il client FL con privacy differenziale
    client = FlowerClientWithDP(model, args.partition_id)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()