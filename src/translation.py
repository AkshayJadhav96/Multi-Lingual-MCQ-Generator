from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import json
import os
import time
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def translate_text(text, src_lang, tgt_lang, model, tokenizer):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def translate_mcqs():
    print("Starting translation...", flush=True)
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directly load mBART model and tokenizer from Hugging Face
    print("Loading mBART model and tokenizer from 'facebook/mbart-large-50-many-to-many-mmt'...", flush=True)
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model.to(device)
    print("Loaded mBART model and tokenizer", flush=True)

    for lang, tgt_lang, src_code, tgt_code in [('en', 'hi', 'en_XX', 'hi_IN'), ('hi', 'en', 'hi_IN', 'en_XX')]:
        input_file = os.path.join(DATA_DIR, f'generated/mcqs_{lang}.json')
        print(f"Translating {lang} MCQs to {tgt_lang} from {input_file}...", flush=True)
        with open(input_file, 'r', encoding='utf-8') as f:
            mcqs = json.load(f)
        
        for i, item in enumerate(mcqs):
            item[f'question_{tgt_lang}'] = translate_text(item['question'], src_code, tgt_code, model, tokenizer)
            item[f'options_{tgt_lang}'] = [translate_text(opt, src_code, tgt_code, model, tokenizer) for opt in item['options']]
            item[f'correct_answer_{tgt_lang}'] = translate_text(item['correct_answer'], src_code, tgt_code, model, tokenizer)
            if i % 100 == 0:
                print(f"Translated {i+1}/{len(mcqs)} MCQs", flush=True)
        
        output_file = os.path.join(DATA_DIR, f'generated/mcqs_{lang}_translated.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mcqs, f, ensure_ascii=False, indent=2)
        print(f"Translated MCQs saved to {output_file}", flush=True)

    print(f"Translation completed in {(time.time() - start_time) / 3600:.2f} hours", flush=True)

if __name__ == "__main__":
    translate_mcqs()