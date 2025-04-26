import torch
import json
import os
import numpy as np
from ak_lstm_encoder import LSTMEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import random
from sentence_transformers import SentenceTransformer
from indicnlp.tokenize.sentence_tokenize import sentence_split
import spacy
from keybert import KeyBERT
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import fasttext

# Ensure NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Domain-specific distractor lists
DOMAIN_DISTRACTORS = {
    'en': {
        'software': ['Spark', 'Flink', 'Kafka', 'Storm'],
        'education': ['teacher', 'parent', 'peer', 'principal']
    },
    'hi': {
        'शिक्षा': ['शिक्षक', 'अभिभावक', 'सहपाठी', 'प्रधानाचार्य'],
        'विद्यार्थी': ['शिक्षक', 'अभिभावक', 'सहपाठी', 'प्रधानाचार्य']
    }
}

# Reference questions for evaluation


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

def extract_answers(context, tokenizer, model, lstm, word2idx, device, lang='en'):
    sents = sentence_split(context, lang) if lang == 'hi' else sent_tokenize(context)
    kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
    try:
        nlp = spacy.load("en_core_web_sm") if lang == 'en' else None
    except OSError:
        print("Warning: SpaCy model 'en_core_web_sm' not found. Falling back to KeyBERT and heuristics.", flush=True)
        nlp = None
    answers = []

    for sent in sents:
        keywords = kw_model.extract_keywords(sent, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=1)
        answer = keywords[0][0] if keywords else None

        if lang == 'en' and nlp:
            doc = nlp(sent)
            entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]]
            if entities:
                answer = entities[0]
            elif not answer:
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                        answer = token.text
                        break

        if not answer:
            tokens = word_tokenize(sent)
            if tokens:
                for i, token in enumerate(tokens):
                    if token not in ["के", "में", "से", "है", "और"] and len(token) > 2:
                        if i + 1 < len(tokens) and tokens[i + 1] in ["छात्र", "विद्यार्थी", "person"]:
                            answer = f"{token} {tokens[i + 1]}"
                            break
                        answer = token
                        break
                else:
                    answer = tokens[0]

        answer = " ".join(answer.split()[:2])
        if not answer:
            answer = "unknown"
        
        print(f"Extracted answer for '{sent}': {answer}", flush=True)
        answers.append(answer)

    print(f"Extracted answers: {answers}", flush=True)
    return sents, answers

def generate_distractors(answer, context, question, lang):
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    distractors = []

    for domain, terms in DOMAIN_DISTRACTORS[lang].items():
        if answer.lower() in domain.lower() or domain.lower() in answer.lower():
            distractors = [term for term in terms if term.lower() != answer.lower()]
            break

    if len(distractors) < 3:
        if lang == 'hi':
            try:
                ft_model = fasttext.load_model(os.path.join(DATA_DIR, 'cc.hi.300.bin'))
                distractors = [word for word, _ in ft_model.get_nearest_neighbors(answer, k=10) if word != answer][:3]
            except Exception as e:
                print(f"Warning: FastText model error: {e}. Falling back to semantic similarity.", flush=True)
        if len(distractors) < 3:
            context_words = word_tokenize(context)
            candidate_phrases = [word for word in context_words if word.lower() != answer.lower() and len(word) > 2]
            if candidate_phrases:
                answer_embedding = encoder.encode(answer, convert_to_tensor=True)
                candidate_embeddings = encoder.encode(candidate_phrases, convert_to_tensor=True)
                similarities = torch.cosine_similarity(answer_embedding, candidate_embeddings, dim=-1)
                sorted_indices = torch.argsort(similarities, descending=True)
                selected_indices = sorted_indices[1:7]
                for idx in selected_indices:
                    distractor = candidate_phrases[idx]
                    if distractor not in distractors and len(distractors) < 3:
                        distractors.append(distractor)

    if len(distractors) < 3:
        fallback = {
            'en': ['option1', 'option2', 'option3'],
            'hi': ['विकल्प1', 'विकल्प2', 'विकल्प3']
        }
        distractors.extend(fallback[lang][:3 - len(distractors)])

    return distractors[:3]

def evaluate_questions(generated_questions, reference_questions, lang):
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    reference_to_use = reference_questions[:min(len(reference_questions), len(generated_questions))]
    while len(reference_to_use) < len(generated_questions):
        reference_to_use.append(reference_questions[-1])

    for gen_q, ref_q in zip(generated_questions, reference_to_use):
        from nltk.translate.bleu_score import SmoothingFunction
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref_q.split()], gen_q.split(), smoothing_function=smooth)
        bleu_scores.append(bleu)
        meteor = meteor_score([ref_q.split()], gen_q.split())
        meteor_scores.append(meteor)
        rouge = scorer.score(ref_q, gen_q)
        rouge_scores.append(rouge['rougeL'].fmeasure)

    return {
        'BLEU': np.mean(bleu_scores),
        'METEOR': np.mean(meteor_scores),
        'ROUGE-L': np.mean(rouge_scores)
    }

def generate_mcq(text, lang='en', translate=False):
    print(f"Generating MCQ for text: {text}", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_lang = lang

    # Translation setup for Hindi
    hi_en_model, hi_en_tokenizer, en_hi_model, en_hi_tokenizer = None, None, None, None
    if translate and lang == 'hi':
        hi_en_model_name = "facebook/m2m100_418M"
        en_hi_model_name = "facebook/m2m100_418M"
        
        hi_en_tokenizer = M2M100Tokenizer.from_pretrained(hi_en_model_name)
        hi_en_tokenizer.src_lang = "hi"
        hi_en_tokenizer.tgt_lang = "en"
        hi_en_model = M2M100ForConditionalGeneration.from_pretrained(hi_en_model_name).to(device)
        
        en_hi_tokenizer = M2M100Tokenizer.from_pretrained(en_hi_model_name)
        en_hi_tokenizer.src_lang = "en"
        en_hi_tokenizer.tgt_lang = "hi"
        en_hi_model = M2M100ForConditionalGeneration.from_pretrained(en_hi_model_name).to(device)

        # Translate Hindi context to English
        inputs = hi_en_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        translated_ids = hi_en_model.generate(
            **inputs, 
            max_length=512, 
            num_beams=4,
            forced_bos_token_id=hi_en_tokenizer.get_lang_id("en")
        )
        english_text = hi_en_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        print(f"Translated Hindi to English: {english_text}", flush=True)
        text = english_text
        lang = 'en'

    # Load embeddings and models
    embedding_file = os.path.join(DATA_DIR, 'embeddings/glove.6B.300d.txt') if lang == 'en' else os.path.join(DATA_DIR, 'embeddings/cc.hi.300.vec')
    embeddings, word2idx = load_embeddings(embedding_file)
    embedding_matrix = torch.stack([embeddings.get(word, torch.zeros(300)) for word in embeddings.keys()])
    
    model_path = os.path.join(MODEL_DIR, 'lstm_english/model.pt') if lang == 'en' else os.path.join(MODEL_DIR, 'lstm_hindi/model.pt')
    lstm = LSTMEncoder(embedding_dim=300, hidden_dim=256, embeddings=embedding_matrix)
    lstm.load_state_dict(torch.load(model_path))
    lstm.to(device)
    lstm.eval()

    transformer_path = os.path.join(MODEL_DIR, 'transformer/finetuned')
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(transformer_path)
    model.to(device)

    # Extract sentences and answers
    sents, answers = extract_answers(text, tokenizer, model, lstm, word2idx, device, lang)
    
    # Translate answers to English for Hindi pipeline
    if translate and original_lang == 'hi':
        translated_answers = []
        for answer in answers:
            inputs = hi_en_tokenizer(answer, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            translated_ids = hi_en_model.generate(
                **inputs, 
                max_length=512, 
                num_beams=4,
                forced_bos_token_id=hi_en_tokenizer.get_lang_id("en")
            )
            translated_answer = hi_en_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            print(f"Translated answer '{answer}' to English: {translated_answer}", flush=True)
            translated_answers.append(translated_answer)
        answers = translated_answers

    mcqs = []
    generated_questions = []

    with torch.no_grad():
        for sent, answer in zip(sents, answers):
            if not answer or answer == "unknown":
                print(f"Skipping invalid answer for '{sent}'", flush=True)
                continue
            sent_words = word_tokenize(sent)
            sent_ids = [word2idx.get(word, 0) for word in sent_words[:512]]
            if not sent_ids:
                sent_ids = [0]
            sent_tensor = torch.tensor([sent_ids], dtype=torch.long).to(device)
            lstm_output = lstm(sent_tensor)
            if lstm_output.dim() == 3:
                lstm_mean = lstm_output.mean(dim=1).squeeze(0)
            elif lstm_output.dim() == 2:
                lstm_mean = lstm_output.squeeze(0)
            else:
                lstm_mean = torch.zeros(256).to(device)
            lstm_mean = lstm_mean.tolist() if torch.is_tensor(lstm_mean) else [lstm_mean] * 256
            lstm_str = " ".join([f"{x:.4f}" for x in lstm_mean[:10]])

            prompt = "generate clear and concise question: "
            input_text = f"{prompt}{sent} answer: {answer} [LSTM: {lstm_str}]"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
            question_ids = model.generate(**inputs, max_length=32, num_beams=4, length_penalty=0.8)
            question = tokenizer.decode(question_ids[0], skip_special_tokens=True)
            # print(f"Generated question: {question}", flush=True)

            # Generate distractors in English
            distractors = generate_distractors(answer, text, question, lang='en')

            # Translate question, answer, and distractors back to Hindi if needed
            if translate and original_lang == 'hi':
                try:
                    # Translate question
                    inputs = en_hi_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                    translated_ids = en_hi_model.generate(
                        **inputs, 
                        max_length=512, 
                        num_beams=4,
                        forced_bos_token_id=en_hi_tokenizer.get_lang_id("hi")
                    )
                    question = en_hi_tokenizer.decode(translated_ids[0], skip_special_tokens=True)

                    # Translate answer
                    inputs = en_hi_tokenizer(answer, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                    translated_ids = en_hi_model.generate(
                        **inputs, 
                        max_length=512, 
                        num_beams=4,
                        forced_bos_token_id=en_hi_tokenizer.get_lang_id("hi")
                    )
                    answer = en_hi_tokenizer.decode(translated_ids[0], skip_special_tokens=True)

                    # Translate distractors
                    translated_distractors = []
                    for distractor in distractors:
                        inputs = en_hi_tokenizer(distractor, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                        translated_ids = en_hi_model.generate(
                            **inputs, 
                            max_length=512, 
                            num_beams=4,
                            forced_bos_token_id=en_hi_tokenizer.get_lang_id("hi")
                        )
                        translated_distractors.append(en_hi_tokenizer.decode(translated_ids[0], skip_special_tokens=True))
                    distractors = translated_distractors
                except Exception as e:
                    print(f"Translation error: {e}", flush=True)

            options = [answer] + distractors
            random.shuffle(options)

            # Clean up Hindi question markers
            if translate and original_lang == 'hi':
                question = question.replace(" क्या है? क्या है?", " क्या है?")
                question = question.replace(" कितने हैं? कितने हैं?", " कितने हैं?")
                question = question.replace(" कौन है? कौन है?", " कौन है?")
                if "?" not in question:
                    if "कौन" in question or "who" in question.lower():
                        question += " कौन है?"
                    elif "कितने" in question or "how many" in question.lower():
                        question += " कितने हैं?"
                    else:
                        question += " क्या है?"

            mcq = {
                "question": question,
                "options": options,
                "correct_answer": answer
            }
            mcqs.append(mcq)
            generated_questions.append(mcq["question"])
            print(json.dumps(mcq, indent=2, ensure_ascii=False), flush=True)

    # Evaluate generated questions
    # eval_metrics = evaluate_questions(generated_questions, REFERENCE_QUESTIONS[original_lang], original_lang)
    # print(f"Evaluation Metrics: {json.dumps(eval_metrics, indent=2)}", flush=True)

    return mcqs

if __name__ == "__main__":
    hindi_text = """पर्यावरण संरक्षण आज के समय में एक महत्वपूर्ण मुद्दा है। पेड़-पौधे और जंगल हमारे ग्रह के लिए ऑक्सीजन का मुख्य स्रोत हैं। नदियाँ और झीलें स्वच्छ जल प्रदान करती हैं, जो जीवन के लिए आवश्यक है। लेकिन प्रदूषण, जैसे कि प्लास्टिक कचरा और औद्योगिक अपशिष्ट, हमारे पर्यावरण को नुकसान पहुँचाता है। भारत में, गंगा नदी को स्वच्छ करने के लिए कई अभियान चलाए गए हैं, जैसे कि 'नमामि गंगे'। हर व्यक्ति को पुनर्चक्रण और ऊर्जा संरक्षण जैसे कदम उठाकर पर्यावरण की रक्षा में योगदान देना चाहिए।"""
    mcqs_hi = generate_mcq(hindi_text, lang='hi', translate=True)
    print(f"Generated {len(mcqs_hi)} MCQs for Hindi", flush=True)

    english_text = """Machine learning, a subset of artificial intelligence, enables computers to learn from data without explicit programming. Supervised learning, one of its core branches, involves training models on labeled datasets, such as predicting house prices using features like size and location. Popular algorithms include linear regression, decision trees, and neural networks, with neural networks powering advanced applications like image recognition. The Python library scikit-learn provides tools for implementing these algorithms, while TensorFlow and PyTorch are widely used for deep learning. Overfitting, a common challenge, occurs when a model learns noise instead of patterns, but techniques like cross-validation and regularization help mitigate it."""
    mcqs_en = generate_mcq(english_text, lang='en', translate=False)
    print(f"Generated {len(mcqs_en)} MCQs for English", flush=True)