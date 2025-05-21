import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import json

# Define the sentiment analysis model
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        # If bidirectional, we need to multiply hidden_dim by 2
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.bidirectional = bidirectional

    def forward(self, text):
        # text shape: [batch size, sent len]
        embedded = self.embedding(text)
        # embedded shape: [batch size, sent len, emb dim]
        output, (hidden, cell) = self.rnn(embedded)

        # If bidirectional, concatenate the final forward and backward hidden states
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        return self.fc(hidden)

# Create a simple dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize (very simple)
        tokens = [self.vocab.get(word, 1) for word in text.lower().split()]

        # Pad or truncate
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return torch.tensor(tokens), torch.tensor(label)

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        text, labels = batch
        text, labels = text.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss / len(dataloader), accuracy

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)

            predictions = model(text)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()

            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss / len(dataloader), accuracy

def main():
    # Create a tiny dataset for sentiment analysis
    # 0 = negative, 1 = positive
    texts = [
        "i love this movie",
        "this film is amazing",
        "great acting and plot",
        "wonderful experience watching this",
        "best movie i've seen this year",
        "i hated this movie",
        "terrible acting and storyline",
        "waste of money and time",
        "disappointing film overall",
        "worst movie ever made",
        "the actors did well but the plot was boring",
        "beautiful cinematography but poor dialogue",
        "the movie started great but ended poorly",
        "it was average at best",
        "not bad but not great either"
    ]

    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Create a simple vocabulary
    words = set()
    for text in texts:
        for word in text.lower().split():
            words.add(word)

    vocab = {'<pad>': 0, '<unk>': 1}
    for i, word in enumerate(words, 2):
        vocab[word] = i

    # Save the vocabulary for later use
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/vocab.json', 'w') as f:
        json.dump(vocab, f)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = SentimentDataset(train_texts, train_labels, vocab)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # Set up model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save model configuration for later use
    model_config = {
        'vocab_size': len(vocab),
        'embedding_dim': 64,
        'hidden_dim': 64,
        'output_dim': 2,  # binary classification
        'bidirectional': True
    }

    with open('data/processed/model_config.json', 'w') as f:
        json.dump(model_config, f)

    # Create model instance
    model = SentimentClassifier(
        model_config['vocab_size'],
        model_config['embedding_dim'],
        model_config['hidden_dim'],
        model_config['output_dim'],
        model_config['bidirectional']
    )
    model = model.to(device)

    # Training parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 100

    # Training loop
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(n_epochs):
        train_loss, train_acc = train_model(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'data/processed/sentiment_model.pt')

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"Epoch: {epoch+1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load the best model
    model.load_state_dict(torch.load('data/processed/sentiment_model.pt'))

    # Try to export to ONNX if possible
    try:
        dummy_input = torch.zeros(1, 50, dtype=torch.long).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            'data/processed/sentiment_model.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to data/processed/sentiment_model.onnx")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Skipping ONNX export. To enable ONNX export, install the 'onnx' package with 'conda install -c conda-forge onnx'")

    print("Training complete!")
    print(f"Model saved to data/processed/sentiment_model.pt")
    print(f"ONNX model saved to data/processed/sentiment_model.onnx")
    print(f"Vocabulary saved to data/processed/vocab.json")
    print(f"Model configuration saved to data/processed/model_config.json")

if __name__ == "__main__":
    main()
