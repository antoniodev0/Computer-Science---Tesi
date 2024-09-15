import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from opacus import PrivacyEngine
import flwr as fl
import warnings
import random
from opacus.layers import DPLSTM
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated")

# Parsing degli argomenti per la partizione del client
parser = argparse.ArgumentParser(description='Federated Learning Client with DP')
parser.add_argument('--partition-id', type=int, required=True, help='Partition ID')
args = parser.parse_args()

# Modello LSTM per Shakespeare
class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(ShakespeareLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = DPLSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = self.fc(out)
        return out

# Dataset personalizzato per Shakespeare
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.vocab = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        seq = self.text[idx:idx+self.seq_length]
        target = self.text[idx+1:idx+self.seq_length+1]
        seq_idx = torch.tensor([self.char_to_idx[c] for c in seq])
        target_idx = torch.tensor([self.char_to_idx[c] for c in target])
        return seq_idx, target_idx

# Funzione per caricare i dati e dividerli tra i client
def load_shakespeare_data(partition_id, num_partitions=5, seq_length=100):
    with open('shakespeare.txt', 'r') as file:
        text = file.read()

    dataset = ShakespeareDataset(text, seq_length)

    # Dividere il dataset in base alla partizione
    partition_size = len(dataset) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size
    partition_dataset = random_split(dataset, [partition_size, len(dataset) - partition_size])[0]

    # Dividere in training e validation set (80% training, 20% validation)
    train_size = int(0.8 * len(partition_dataset))
    val_size = len(partition_dataset) - train_size
    train_dataset, val_dataset = random_split(partition_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, dataset.vocab

# Client FL con DP
class FlowerClientWithDP(fl.client.NumPyClient):
    def __init__(self, model, partition_id, num_partitions=100, delta=1e-5):
        self.model = model
        self.partition_id = partition_id
        self.train_loader, self.val_loader, _ = load_shakespeare_data(partition_id, num_partitions)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Inizializza il PrivacyEngine
        self.privacy_engine = PrivacyEngine()

        # Collegamento del PrivacyEngine al modello e all'ottimizzatore
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=0.8,  # Controllo del rumore
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
        
        # Numero totale di batch in tutte le epoche
        total_batches = 3 * len(self.train_loader)
        
        # Inizializza la barra di progressione
        pbar = tqdm(total=total_batches, desc=f"Training (Client {self.partition_id})", unit="batch")
        
        try:
            for _ in range(3):  # 5 epoche
                for inputs, labels in self.train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    outputs = outputs.view(-1, outputs.size(2))
                    labels = labels.view(-1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    # Aggiorna la barra di progressione
                    pbar.update(1)
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        finally:
            pbar.close()

        # Calcolo dell'epsilon dopo l'addestramento
        epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
        print(f"Training complete. (ε = {epsilon:.2f}, δ = {self.delta})")

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(2))
                labels = labels.view(-1)
                loss += self.criterion(outputs, labels).item() * len(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        avg_loss = loss / len(self.val_loader.dataset)
        return avg_loss, len(self.val_loader.dataset), {"loss": avg_loss, "accuracy": accuracy}

# Funzione principale per lanciare il client
def main():
    # Inizializza il modello LSTM
    model = ShakespeareLSTM(vocab_size=256, embed_size=64, hidden_size=128, num_layers=2) 

    # Crea il client FL con privacy differenziale
    client = FlowerClientWithDP(model, args.partition_id)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()
