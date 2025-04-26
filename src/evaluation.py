import json
import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
GENERATED_DIR = os.path.join(DATA_DIR, 'generated')
OUTPUT_DIR = os.path.join(DATA_DIR, 'evaluation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_questions(generated_questions, reference_questions, lang):
    """Evaluate generated questions against reference questions using BLEU, METEOR, and ROUGE-L."""
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    # Pair generated and reference questions
    min_length = min(len(generated_questions), len(reference_questions))
    if min_length == 0:
        print(f"Warning: No questions to evaluate for {lang}", flush=True)
        return {'BLEU': 0.0, 'METEOR': 0.0, 'ROUGE-L': 0.0}

    for i in range(min_length):
        gen_q = generated_questions[i]['question']
        ref_q = reference_questions[i]['question']

        # BLEU with smoothing
        bleu = sentence_bleu([ref_q.split()], gen_q.split(), smoothing_function=smooth)
        bleu_scores.append(bleu)

        # METEOR
        meteor = meteor_score([ref_q.split()], gen_q.split())
        meteor_scores.append(meteor)

        # ROUGE-L
        rouge = scorer.score(ref_q, gen_q)
        rouge_scores.append(rouge['rougeL'].fmeasure)

    # Handle remaining questions by repeating the last reference
    for i in range(min_length, len(generated_questions)):
        gen_q = generated_questions[i]['question']
        ref_q = reference_questions[-1]['question'] if reference_questions else ""

        bleu = sentence_bleu([ref_q.split()], gen_q.split(), smoothing_function=smooth)
        bleu_scores.append(bleu)

        meteor = meteor_score([ref_q.split()], gen_q.split())
        meteor_scores.append(meteor)

        rouge = scorer.score(ref_q, gen_q)
        rouge_scores.append(rouge['rougeL'].fmeasure)

    return {
        'BLEU': np.mean(bleu_scores) if bleu_scores else 0.0,
        'METEOR': np.mean(meteor_scores) if meteor_scores else 0.0,
        'ROUGE-L': np.mean(rouge_scores) if rouge_scores else 0.0
    }

def evaluate(lang='en'):
    """Evaluate generated questions for a given language."""
    print(f"Starting evaluation for {lang}...", flush=True)

    # Load generated questions
    generated_file = os.path.join(GENERATED_DIR, f'questions_{lang}.json')
    try:
        generated_questions = load_json_file(generated_file)
    except FileNotFoundError:
        print(f"Error: Generated questions file {generated_file} not found", flush=True)
        return None

    # Load reference dataset
    reference_file = os.path.join(DATA_DIR, 'processed', f'{lang}_processed.json')
    try:
        reference_data = load_json_file(reference_file)
    except FileNotFoundError:
        print(f"Error: Reference dataset {reference_file} not found", flush=True)
        return None

    # Evaluate
    metrics = evaluate_questions(generated_questions, reference_data, lang)
    print(f"Evaluation metrics for {lang}: {json.dumps(metrics, indent=2)}", flush=True)

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f'evaluation_{lang}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_file}", flush=True)

    return metrics

if __name__ == "__main__":
    # Evaluate English questions
    en_metrics = evaluate('en')

    # Evaluate Hindi questions
    hi_metrics = evaluate('hi')

    # Combine results
    combined_results = {
        'english': en_metrics or {'BLEU': 0.0, 'METEOR': 0.0, 'ROUGE-L': 0.0},
        'hindi': hi_metrics or {'BLEU': 0.0, 'METEOR': 0.0, 'ROUGE-L': 0.0}
    }
    combined_output_file = os.path.join(OUTPUT_DIR, 'evaluation_combined.json')
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)
    print(f"Combined evaluation results saved to {combined_output_file}", flush=True)