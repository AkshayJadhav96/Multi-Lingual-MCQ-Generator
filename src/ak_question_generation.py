import torch
import json
import os
import numpy as np
from ak_lstm_encoder import LSTMEncoder, TextDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk import sent_tokenize
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def load_embeddings(file_path, vocab_size=10000):
    print(f"Loading embeddings from {file_path}...", flush=True)
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= vocab_size: break
            tokens = line.strip().split()
            if len(tokens) != 301: continue
            embeddings[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    print(f"Loaded {len(embeddings)} embeddings", flush=True)
    return embeddings

def extract_answers(context, tokenizer, model, device):
    sents = sent_tokenize(context)
    inputs = [f"extract answers: {sent}" for sent in sents]
    tokenized = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outs = model.generate(
        input_ids=tokenized['input_ids'].to(device),
        attention_mask=tokenized['attention_mask'].to(device),
        max_length=32
    )
    answers = [tokenizer.decode(out, skip_special_tokens=True).split('<sep>')[0] for out in outs]
    return sents, answers

def generate_questions(lang='en'):
    print(f"Starting question generation for {lang}...", flush=True)
    start_time = time.time()

    embedding_file = os.path.join(DATA_DIR, 'embeddings/glove.6B.300d.txt') if lang == 'en' else os.path.join(DATA_DIR, 'embeddings/cc.hi.300.vec')
    data_file = os.path.join(DATA_DIR, 'processed/english_processed.json') if lang == 'en' else os.path.join(DATA_DIR, 'processed/hindi_train.json')
    model_path = os.path.join(MODEL_DIR, 'lstm_english/model.pt') if lang == 'en' else os.path.join(MODEL_DIR, 'lstm_hindi/model.pt')
    transformer_path = os.path.join(MODEL_DIR, 'transformer/finetuned')

    embeddings = load_embeddings(embedding_file)
    embedding_matrix = np.array([embeddings[word] for word in embeddings.keys()])

    print(f"Loading dataset from {data_file}...", flush=True)
    dataset = TextDataset(data_file, embeddings, lang=lang)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset loaded with {len(dataset)} samples", flush=True)

    print(f"Loading LSTM model from {model_path}...", flush=True)
    if not os.path.exists(model_path):
        print(f"Error: LSTM model path {model_path} does not exist.", flush=True)
        raise FileNotFoundError(f"LSTM model path {model_path} not found")
    lstm_model = LSTMEncoder(embedding_dim=300, hidden_dim=256, embeddings=embedding_matrix)
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model.eval()

    print(f"Loading fine-tuned transformer from {transformer_path}...", flush=True)
    if not os.path.exists(transformer_path):
        print(f"Error: Fine-tuned transformer path {transformer_path} does not exist.", flush=True)
        raise FileNotFoundError(f"Transformer path {transformer_path} not found")
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    transformer = AutoModelForSeq2SeqLM.from_pretrained(transformer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)

    questions = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Processing batch {i+1}/{len(loader)}...", flush=True)
            batch_size = batch.size(0)
            for j in range(batch_size):
                idx = i * 32 + j
                if idx >= len(dataset):
                    break
                context = dataset.data[idx]['context']
                sents, answers = extract_answers(context, tokenizer, transformer, device)
                for sent, answer in zip(sents, answers):
                    if not answer or len(answer.strip()) < 2:
                        continue
                    input_text = f"generate question: <hl> {answer} <hl> {context}"
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
                    question_ids = transformer.generate(
                        input_ids=inputs['input_ids'].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        max_length=32,
                        num_beams=4
                    )
                    question = tokenizer.decode(question_ids[0], skip_special_tokens=True)
                    questions.append({"context": context, "question": question, "answer": answer})
            print(f"Generated questions for batch {i+1}/{len(loader)}", flush=True)

    os.makedirs(os.path.join(DATA_DIR, 'generated'), exist_ok=True)
    output_file = os.path.join(DATA_DIR, f'generated/questions_{lang}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Generated questions saved to {output_file} in {(time.time() - start_time) / 60:.2f} minutes", flush=True)

if __name__ == "__main__":
    generate_questions('en')
    generate_questions('hi')