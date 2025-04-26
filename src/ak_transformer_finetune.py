import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
import json
import os
import time
import random
import nltk
from ak_lstm_encoder import LSTMEncoder
from nltk.tokenize import word_tokenize

nltk.data.path.append('/home/u142201016/nltk_data')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def load_data(file_path):
    print(f"Loading data from {file_path}...", flush=True)
    start = time.time()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples in {time.time() - start:.2f}s", flush=True)
    return data

def load_embeddings(file_path, vocab_size=10000):
    embeddings = {}
    word2idx = {}
    idx = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= vocab_size: break
            tokens = line.strip().split()
            if len(tokens) != 301: continue
            word = tokens[0]
            embeddings[word] = torch.tensor([float(x) for x in tokens[1:]])
            word2idx[word] = idx
            idx += 1
    return embeddings, word2idx

class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, lstm_en, lstm_hi, word2idx_en, word2idx_hi, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.lstm_en = lstm_en
        self.lstm_hi = lstm_hi
        self.word2idx_en = word2idx_en
        self.word2idx_hi = word2idx_hi
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']
        lang = 'hi' if any('\u0900' <= c <= '\u097f' for c in context) else 'en'
        lstm = self.lstm_hi if lang == 'hi' else self.lstm_en
        word2idx = self.word2idx_hi if lang == 'hi' else self.word2idx_en

        # Tokenize context for LSTM
        context_words = word_tokenize(context)
        context_ids = [word2idx.get(word, 0) for word in context_words[:self.max_length]]
        if not context_ids:
            context_ids = [0]  # Default to index 0 if empty
        context_tensor = torch.tensor([context_ids], dtype=torch.long)
        with torch.no_grad():
            lstm_output = lstm(context_tensor)  # [1, seq_len, hidden_dim]
            if lstm_output.dim() == 3:  # Ensure [batch, seq_len, hidden_dim]
                lstm_mean = lstm_output.mean(dim=1).squeeze(0)  # [hidden_dim]
            elif lstm_output.dim() == 2:  # [batch, hidden_dim] if seq_len=1
                lstm_mean = lstm_output.squeeze(0)  # [hidden_dim]
            else:  # Scalar or unexpected shape
                lstm_mean = torch.zeros(256)  # Fallback to zero vector
            lstm_mean = lstm_mean.tolist() if torch.is_tensor(lstm_mean) else [lstm_mean] * 256
        lstm_str = " ".join([f"{x:.4f}" for x in lstm_mean[:10]])

        if random.random() < 0.5:
            input_text = f"answer: {context} [LSTM: {lstm_str}]"
            target_text = answer
        else:
            input_text = f"generate question: {context} answer: {answer} [LSTM: {lstm_str}]"
            target_text = question

        inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        outputs = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": outputs["input_ids"].squeeze()
        }

def fine_tune_transformer():
    print("Starting transformer fine-tuning...", flush=True)
    start_time = time.time()

    # Directly load mT5 model and tokenizer from Hugging Face
    print("Loading mT5 model and tokenizer from 'google/mt5-small'...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

    # Load English LSTM
    embedding_file_en = os.path.join(DATA_DIR, 'embeddings/glove.6B.300d.txt')
    embeddings_en, word2idx_en = load_embeddings(embedding_file_en)
    embedding_matrix_en = torch.stack([embeddings_en.get(word, torch.zeros(300)) for word in embeddings_en.keys()])
    lstm_path_en = os.path.join(MODEL_DIR, 'lstm_english/model.pt')
    lstm_en = LSTMEncoder(embedding_dim=300, hidden_dim=256, embeddings=embedding_matrix_en)
    lstm_en.load_state_dict(torch.load(lstm_path_en))
    lstm_en.eval()

    # Load Hindi LSTM
    embedding_file_hi = os.path.join(DATA_DIR, 'embeddings/cc.hi.300.vec')
    embeddings_hi, word2idx_hi = load_embeddings(embedding_file_hi)
    embedding_matrix_hi = torch.stack([embeddings_hi.get(word, torch.zeros(300)) for word in embeddings_hi.keys()])
    lstm_path_hi = os.path.join(MODEL_DIR, 'lstm_hindi/model.pt')
    lstm_hi = LSTMEncoder(embedding_dim=300, hidden_dim=256, embeddings=embedding_matrix_hi)
    lstm_hi.load_state_dict(torch.load(lstm_path_hi))
    lstm_hi.eval()

    print("Loaded mT5 model, tokenizer, and LSTMs", flush=True)

    english_data = load_data(os.path.join(DATA_DIR, 'processed/english_processed.json'))
    hindi_data = load_data(os.path.join(DATA_DIR, 'processed/hindi_processed.json'))
    combined_data = english_data + hindi_data
    dataset = QADataset(combined_data, tokenizer, lstm_en, lstm_hi, word2idx_en, word2idx_hi)
    print(f"Prepared dataset with {len(dataset)} samples", flush=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, 'transformer/finetuned'),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Starting training...", flush=True)
    trainer.train()
    print(f"Training completed in {(time.time() - start_time) / 3600:.2f} hours", flush=True)

    finetuned_path = os.path.join(MODEL_DIR, 'transformer/finetuned')
    print(f"Saving fine-tuned model to {finetuned_path}...", flush=True)
    os.makedirs(finetuned_path, exist_ok=True)
    model.save_pretrained(finetuned_path)
    tokenizer.save_pretrained(finetuned_path)
    print(f"Fine-tuned transformer saved to {finetuned_path}", flush=True)

if __name__ == "__main__":
    fine_tune_transformer()