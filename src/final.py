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
REFERENCE_QUESTIONS = {
    'hi': [
        "शिक्षा में अच्छा कौन होता है?",
        "उसका दोस्त कौन होना चाहता है?",
        "आदर्श छात्र का मुख्य गुण क्या है?"
    ],
    'en': [
        "What is an open source framework for distributed storage?",
        "What is Hadoop designed to scale up to?",
        "What does Hadoop process efficiently?"
    ]
}

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
        # Use KeyBERT with preference for short phrases
        keywords = kw_model.extract_keywords(sent, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=1)
        answer = keywords[0][0] if keywords else None

        # Use SpaCy for NER and POS (English only, if available)
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

        # Fallback for Hindi or if no answer found
        if not answer:
            tokens = word_tokenize(sent)
            if tokens:
                # Prioritize compound nouns or significant terms
                for i, token in enumerate(tokens):
                    if token not in ["के", "में", "से", "है", "और"] and len(token) > 2:
                        # Check for compound nouns (e.g., "आदर्श छात्र")
                        if i + 1 < len(tokens) and tokens[i + 1] in ["छात्र", "विद्यार्थी", "person"]:
                            answer = f"{token} {tokens[i + 1]}"
                            break
                        answer = token
                        break
                else:
                    answer = tokens[0]

        # Ensure answer is concise (1-2 words)
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

    # Try domain-specific distractors first
    for domain, terms in DOMAIN_DISTRACTORS[lang].items():
        if answer.lower() in domain.lower() or domain.lower() in answer.lower():
            distractors = [term for term in terms if term.lower() != answer.lower()]
            break

    # If not enough distractors, use FastText for Hindi or semantic similarity for English
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
                selected_indices = sorted_indices[1:7]  # Get more candidates to have variety
                for idx in selected_indices:
                    distractor = candidate_phrases[idx]
                    if distractor not in distractors and len(distractors) < 3:
                        distractors.append(distractor)

    # Fallback: Use generic terms
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
    
    # Use available reference questions
    reference_to_use = reference_questions[:min(len(reference_questions), len(generated_questions))]
    
    # If not enough reference questions, repeat the last one
    while len(reference_to_use) < len(generated_questions):
        reference_to_use.append(reference_questions[-1])

    for gen_q, ref_q in zip(generated_questions, reference_to_use):
        # BLEU with smoothing
        from nltk.translate.bleu_score import SmoothingFunction
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref_q.split()], gen_q.split(), smoothing_function=smooth)
        bleu_scores.append(bleu)
        
        # METEOR
        meteor = meteor_score([ref_q.split()], gen_q.split())
        meteor_scores.append(meteor)
        
        # ROUGE-L
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

    if translate and lang == 'hi':
        hi_en_model_name = "facebook/m2m100_418M"
        en_hi_model_name = "facebook/m2m100_418M"
        
        # Configure Hindi to English translation properly
        hi_en_tokenizer = M2M100Tokenizer.from_pretrained(hi_en_model_name)
        hi_en_tokenizer.src_lang = "hi"
        hi_en_tokenizer.tgt_lang = "en"  # IMPORTANT: Explicitly set target language
        hi_en_model = M2M100ForConditionalGeneration.from_pretrained(hi_en_model_name).to(device)
        
        # Configure English to Hindi translation properly
        en_hi_tokenizer = M2M100Tokenizer.from_pretrained(en_hi_model_name)
        en_hi_tokenizer.src_lang = "en"
        en_hi_tokenizer.tgt_lang = "hi"  # IMPORTANT: Explicitly set target language
        en_hi_model = M2M100ForConditionalGeneration.from_pretrained(en_hi_model_name).to(device)

        # Translate Hindi to English with forced language token
        inputs = hi_en_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        translated_ids = hi_en_model.generate(
            **inputs, 
            max_length=512, 
            num_beams=4,
            forced_bos_token_id=hi_en_tokenizer.get_lang_id("en")  # Force English output
        )
        english_text = hi_en_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        print(f"Translated Hindi to English: {english_text}", flush=True)
        text = english_text
        lang = 'en'

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

    sents, answers = extract_answers(text, tokenizer, model, lstm, word2idx, device, lang)
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

            prompt = "generate clear and concise question: " if lang == 'en' else "स्पष्ट और संक्षिप्त प्रश्न उत्पन्न करें: "
            input_text = f"{prompt}{sent} answer: {answer} [LSTM: {lstm_str}]"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
            question_ids = model.generate(**inputs, max_length=32, num_beams=4, length_penalty=0.8)
            question = tokenizer.decode(question_ids[0], skip_special_tokens=True)
            print(f"Generated question: {question}", flush=True)

            distractors = generate_distractors(answer, text, question, lang)
            options = [answer] + distractors
            random.shuffle(options)

            mcq = {
                "question": question,
                "options": options,
                "correct_answer": answer
            }

            # Translation back to Hindi if needed
            if translate and lang == 'en':
                try:
                    # Translate the question
                    inputs = en_hi_tokenizer(mcq["question"], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                    translated_ids = en_hi_model.generate(
                        **inputs, 
                        max_length=512, 
                        num_beams=4,
                        forced_bos_token_id=en_hi_tokenizer.get_lang_id("hi")  # Force Hindi output
                    )
                    mcq["question"] = en_hi_tokenizer.decode(translated_ids[0], skip_special_tokens=True)

                    # Translate the options
                    translated_options = []
                    for opt in mcq["options"]:
                        inputs = en_hi_tokenizer(opt, return_tensors="pt", truncation=True)  # Fixed syntax error here
                        translated_ids = en_hi_model.generate(
                            **inputs, 
                            max_length=512, 
                            num_beams=4,
                            forced_bos_token_id=en_hi_tokenizer.get_lang_id("hi")  # Force Hindi output
                        )
                        translated_options.append(en_hi_tokenizer.decode(translated_ids[0], skip_special_tokens=True))
                    mcq["options"] = translated_options

                    # Translate the correct answer
                    inputs = en_hi_tokenizer(mcq["correct_answer"], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                    translated_ids = en_hi_model.generate(
                        **inputs, 
                        max_length=512, 
                        num_beams=4,
                        forced_bos_token_id=en_hi_tokenizer.get_lang_id("hi")  # Force Hindi output
                    )
                    mcq["correct_answer"] = en_hi_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
                except Exception as e:
                    print(f"Translation error: {e}", flush=True)

            # Clean up Hindi question markers if needed
            if lang == 'en' and translate:
                # First remove any duplicated question markers
                mcq["question"] = mcq["question"].replace(" क्या है? क्या है?", " क्या है?")
                mcq["question"] = mcq["question"].replace(" कितने हैं? कितने हैं?", " कितने हैं?")
                mcq["question"] = mcq["question"].replace(" कौन है? कौन है?", " कौन है?")
                
                # Only add question marker if the question doesn't already have one
                if "?" not in mcq["question"]:
                    if "कौन" in mcq["question"] or "who" in question.lower():
                        mcq["question"] += " कौन है?"
                    elif "कितने" in mcq["question"] or "how many" in question.lower():
                        mcq["question"] += " कितने हैं?"
                    else:
                        mcq["question"] += " क्या है?"

            mcqs.append(mcq)
            generated_questions.append(mcq["question"])
            print(json.dumps(mcq, indent=2, ensure_ascii=False), flush=True)

    # Evaluate generated questions
    eval_metrics = evaluate_questions(generated_questions, REFERENCE_QUESTIONS[lang], lang)
    print(f"Evaluation Metrics: {json.dumps(eval_metrics, indent=2)}", flush=True)

    return mcqs

if __name__ == "__main__":
    hindi_text = """एक आदर्श छात्र शिक्षा में अच्छा होता है, अतिरिक्त पाठ्यक्रम गतिविधियों में भाग लेता है, अच्छा व्यवहार करता है और सुंदर एवं सुशील दिखता है। हर व्यक्ति उसकी तरह बनना चाहता है और हर व्यक्ति उसका दोस्त बनना चाहता है। शिक्षक भी ऐसे छात्रों को पसंद करते हैं और वे जहां भी जाते हैं उनकी सराहना की जाती है। सभी इस तरह के छात्र के साथ बैठना या मित्रता करना चाहते हैं, तो कई लोग उनके लिए अच्छे की दुआ नहीं करते क्योंकि वे उनसे ईर्ष्या करते हैं। फिर भी इससे आदर्श छात्र के मन को कोई फर्क नहीं पड़ता क्योंकि वह अपने व्यक्तित्व से जीवन में उच्च चीजें हासिल करता है। आदर्श छात्र कोई ऐसा व्यक्ति नहीं है जो उत्तम है और प्रत्येक परीक्षा में पूर्ण अंक प्राप्त करता है या प्रत्येक खेल गतिविधियों, जिसमें वह भाग लेता है, और पदक जीतता है। आदर्श छात्र वह है जो अनुशासित रहता है और जीवन में एक सकारात्मक दृष्टि रखता है।"""
    mcqs_hi = generate_mcq(hindi_text, lang='hi', translate=True)
    print(f"Generated {len(mcqs_hi)} MCQs for Hindi", flush=True)

    english_text = """Apache Hadoop software is an open source framework that allows for the distributed storage and processing of large datasets across clusters of computers using simple programming models. Hadoop is designed to scale up from a single computer to thousands of clustered computers, with each machine offering local computation and storage. In this way, Hadoop can efficiently store and process large datasets ranging in size from gigabytes to petabytes of data."""
    mcqs_en = generate_mcq(english_text, lang='en', translate=False)
    print(f"Generated {len(mcqs_en)} MCQs for English", flush=True)