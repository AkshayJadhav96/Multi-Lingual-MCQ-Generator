import json
import os
import nltk
import spacy
import stanza
from indicnlp.tokenize import indic_tokenize
import numpy as np
from transformers import pipeline

# Download NLTK punkt if not already done
nltk.download('punkt')

# Load spaCy (English) and Stanza (Hindi) models
nlp_en = spacy.load("en_core_web_sm")

# Download Hindi hdtb package for tokenization only
stanza.download('hi', package='hdtb', processors='tokenize')
# Load Stanza (Hindi) model for tokenization only
nlp_hi_tokenize = stanza.Pipeline(lang='hi', processors='tokenize', package='hdtb')

# Load IndicNER model
model_name = "ai4bharat/indicner"
nlp_hi_ner = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to load text files line by line (kept for compatibility)
def load_txt_file(file_path):
    """Load a text file and return a list of lines."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_english_squad(input_file='squad/train-v2.0.json', output_file='english_processed.json'):
    """Preprocess SQuAD dataset for English."""
    input_path = os.path.join(DATA_DIR, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"SQuAD file {input_path} not found")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)['data']

    processed_data = []
    for article in squad_data:
        for para in article['paragraphs']:
            context = para['context']
            doc = nlp_en(context)
            sentences = [sent.text for sent in doc.sents]
            entities = {ent.text: ent.label_ for ent in doc.ents}

            for qa in para['qas']:
                if qa['answers']:  # Only process answerable questions
                    question = qa['question']
                    answer = qa['answers'][0]['text']
                    processed_data.append({
                        'context': context,
                        'sentences': sentences,
                        'entities': entities,
                        'question': question,
                        'answer': answer
                    })

    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"Processed English data saved to {output_file}")

def preprocess_hindi_data(input_file='hindi_data/indicqa.hi.json', output_file='hindi_processed.json'):
    """
    Preprocess Hindi data from indicqa.hi.json.
    
    Args:
        input_file (str): Input JSON file (e.g., 'indicqa.hi.json').
        output_file (str): Output JSON filename (e.g., 'hindi_processed.json').
    """
    input_path = os.path.join(DATA_DIR, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found")

    with open(input_path, 'r', encoding='utf-8') as f:
        hindi_data = json.load(f)['data']

    processed_data = []
    for article in hindi_data:
        for para in article['paragraphs']:
            context = para['context']
            doc = nlp_hi_tokenize(context)
            
            # Tokenize the context into sentences
            sentences = [sent.text for sent in doc.sentences]
            
            # Extract entities using IndicNER
            ner_results = nlp_hi_ner(context)
            entities = {entity['word']: entity['entity_group'] for entity in ner_results}

            for qa in para['qas']:
                if qa['answers'] and qa['answers'][0]['text']:  # Only process answerable questions
                    question = qa['question']
                    answer = qa['answers'][0]['text']
                    processed_data.append({
                        'context': context,
                        'sentences': sentences,
                        'entities': entities,
                        'question': question,
                        'answer': answer
                    })

    # Save the processed data to a JSON file
    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"Processed Hindi data saved to {output_file}")

def load_embeddings(file_path, vocab_size=10000):
    """Load pre-trained embeddings (GloVe or FastText)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file {file_path} not found")
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= vocab_size: break
            tokens = line.strip().split()
            word, vector = tokens[0], [float(x) for x in tokens[1:]]
            embeddings[word] = np.array(vector)
    return embeddings

if __name__ == "__main__":
    # Preprocess English dataset
    preprocess_english_squad()

    # Preprocess Hindi dataset
    preprocess_hindi_data(
        input_file='hindi_data/indicqa.hi.json',
        output_file='hindi_processed.json'
    )

    # Load embeddings (place these files in data/embeddings/)
    glove_embeddings = load_embeddings(os.path.join(DATA_DIR, 'embeddings/glove.6B.300d.txt'))
    fasttext_embeddings = load_embeddings(os.path.join(DATA_DIR, 'embeddings/cc.hi.300.vec'))
    print("Embeddings loaded successfully")