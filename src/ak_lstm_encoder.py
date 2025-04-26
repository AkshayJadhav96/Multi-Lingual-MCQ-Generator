import torch
import torch.nn as nn
import json
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nltk
from indicnlp.tokenize import indic_tokenize
import time

# Download NLTK punkt if not already done
nltk.download('punkt')
nltk.download('punkt_tab')
# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# LSTM Encoder Model
class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, embeddings=None, vocab_size=10000):
        super(LSTMEncoder, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return torch.cat((hidden[0], hidden[1]), dim=-1)  # [batch_size, hidden_dim * 2]

# Dataset class for loading preprocessed JSON data
class TextDataset(Dataset):
    def __init__(self, data_file, embeddings, lang='en', max_length=512):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.embeddings = embeddings
        self.lang = lang
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.build_vocab()

    def build_vocab(self):
        for item in self.data:
            context = item['context'].lower()
            words = self.tokenize(context)
            for word in words:
                if word not in self.word2idx and word in self.embeddings:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def tokenize(self, text):
        if self.lang == 'en':
            return nltk.word_tokenize(text)
        else:
            return indic_tokenize.trivial_tokenize(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context = self.data[idx]['context'].lower()
        words = self.tokenize(context)
        sequence = [self.word2idx.get(word, 1) for word in words]
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        return torch.tensor(sequence, dtype=torch.long)

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

def load_embeddings(file_path, vocab_size=10000):
    """Load pre-trained embeddings (GloVe or FastText) with validation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file {file_path} not found")
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= vocab_size: break
            tokens = line.strip().split()
            if len(tokens) != 301:  # Expecting word + 300 floats
                print(f"Skipping malformed line {i+1} in {file_path}: {len(tokens)} tokens")
                continue
            word, vector = tokens[0], [float(x) for x in tokens[1:]]
            if len(vector) != 300:
                print(f"Skipping line {i+1} with vector length {len(vector)}")
                continue
            embeddings[word] = np.array(vector)
    return embeddings

def train_lstm(model, data_loader, epochs=10, model_name='lstm_english'):
    """Train the LSTM model with progress tracking and time estimation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Dummy loss; replace with a meaningful task later
    
    total_batches = len(data_loader) * epochs
    batches_completed = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        for i, batch in enumerate(data_loader):
            batch_start_time = time.time()
            # No explicit .to(device) since cluster manages GPU automatically
            optimizer.zero_grad()
            output = model(batch)
            target = torch.zeros_like(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            batches_completed += 1
            batch_time = time.time() - batch_start_time
            progress = (batches_completed / total_batches) * 100
            avg_batch_time = (time.time() - epoch_start_time) / (i + 1)
            remaining_batches = total_batches - batches_completed
            eta = remaining_batches * avg_batch_time / 3600  # ETA in hours

            print(
                f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(data_loader)}, "
                f"Loss: {loss.item():.4f}, Progress: {progress:.2f}%, "
                f"ETA: {eta:.2f} hours"
            )

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(data_loader)
        print(
            f"Epoch {epoch+1}/{epochs} completed in {epoch_time/60:.2f} minutes, "
            f"Average Loss: {avg_loss:.4f}"
        )
    
    os.makedirs(os.path.join(MODEL_DIR, model_name), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}/model.pt"))
    print(f"Saved {model_name} to models/{model_name}/model.pt")

if __name__ == "__main__":
    # Load embeddings
    glove_embeddings = load_embeddings(os.path.join(DATA_DIR, 'embeddings/glove.6B.300d.txt'))
    fasttext_embeddings = load_embeddings(os.path.join(DATA_DIR, 'embeddings/cc.hi.300.vec'))

    # Prepare embedding matrices
    glove_matrix = np.array([glove_embeddings[word] for word in glove_embeddings.keys()])
    fasttext_matrix = np.array([fasttext_embeddings[word] for word in fasttext_embeddings.keys()])

    # English LSTM
    english_data = os.path.join(DATA_DIR, 'processed/english_processed.json')
    english_dataset = TextDataset(english_data, glove_embeddings, lang='en')
    english_loader = DataLoader(english_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    lstm_en = LSTMEncoder(embedding_dim=300, hidden_dim=256, embeddings=glove_matrix)
    train_lstm(lstm_en, english_loader, model_name='lstm_english')

    # Hindi LSTM
    hindi_data = os.path.join(DATA_DIR, 'processed/hindi_train.json')
    hindi_dataset = TextDataset(hindi_data, fasttext_embeddings, lang='hi')
    hindi_loader = DataLoader(hindi_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    lstm_hi = LSTMEncoder(embedding_dim=300, hidden_dim=256, embeddings=fasttext_matrix)
    train_lstm(lstm_hi, hindi_loader, model_name='lstm_hindi')