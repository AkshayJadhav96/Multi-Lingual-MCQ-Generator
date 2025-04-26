import torch
import json
import os
import numpy as np
from lstm_encoder import LSTMEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import random
from distractor_generation import generate_distractors
# *********************Akshay**********************
from sentence_transformers import SentenceTransformer
from indicnlp.tokenize.sentence_tokenize import sentence_split 
# *********************Akshay**********************



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

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
    # *********************Akshay**********************
    # sents = sent_tokenize(context)
    sents = sentence_split(context,lang) if lang=='hi' else sent_tokenize(context) #akshay
    # *********************Akshay********************** 
    prompt = "answer: " if lang == 'en' else "उत्तर: "
    answers = []
    for sent in sents:
        sent_words = word_tokenize(sent)
        sent_ids = [word2idx.get(word, 0) for word in sent_words[:512]]
        if not sent_ids:
            sent_ids = [0]
        sent_tensor = torch.tensor([sent_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            lstm_output = lstm(sent_tensor)
            if lstm_output.dim() == 3:
                lstm_mean = lstm_output.mean(dim=1).squeeze(0)
            elif lstm_output.dim() == 2:
                lstm_mean = lstm_output.squeeze(0)
            else:
                lstm_mean = torch.zeros(256).to(device)
            lstm_mean = lstm_mean.tolist() if torch.is_tensor(lstm_mean) else [lstm_mean] * 256
        lstm_str = " ".join([f"{x:.4f}" for x in lstm_mean[:10]])
        input_text = f"{prompt}{sent} [LSTM: {lstm_str}]"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        outs = model.generate(**inputs, max_length=32, num_beams=4)
        raw_answer = tokenizer.decode(outs[0], skip_special_tokens=True).strip()
        # print(f"Raw answer for '{sent}': {raw_answer}", flush=True)  # Debug
        answer = raw_answer
        # if lang == 'hi':
        #     if answer.startswith("उत्तर: "):
        #         answer = answer.replace("उत्तर: ", "").strip()
        #     if "है" in answer and len(answer.split()) > 1:
        #         answer_parts = answer.split(" है")[0].split()
        #         if "नई" in answer or "गंगा" in answer:
        #             answer = " ".join(answer_parts[:2])
        #         elif answer_parts[-1].isdigit():
        #             answer = answer_parts[-1]
        #         else:
        #             answer = answer_parts[0]
        answers.append(answer)
    print(f"Extracted answers: {answers}", flush=True)
    return sents, answers

def generate_mcq(text, lang='en'):
    print(f"Generating MCQ for text: {text}", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # *********************Akshay**********************
    embedding_file = os.path.join(DATA_DIR, 'embeddings/glove.6B.300d.txt') if lang == 'en' else os.path.join(DATA_DIR, 'embeddings/cc.hi.300.vec')
    embeddings, word2idx = load_embeddings(embedding_file)
    embedding_matrix = torch.stack([embeddings.get(word, torch.zeros(300)) for word in embeddings.keys()])
    
    model_path = os.path.join(MODEL_DIR, 'lstm_english/model.pt') if lang == 'en' else os.path.join(MODEL_DIR, 'lstm_hindi/model.pt')
    lstm = LSTMEncoder(embedding_dim=300, hidden_dim=256, embeddings=embedding_matrix, word2idx=word2idx)
    lstm.load_state_dict(torch.load(model_path))
    lstm.to(device)
    lstm.eval()

    # sentence_encoder = SentenceTransformer('sentence-transformers/LaBSE')
    # *********************Akshay**********************

    transformer_path = os.path.join(MODEL_DIR, 'transformer/finetuned')
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(transformer_path)
    model.to(device)

    sents, answers = extract_answers(text, tokenizer, model, lstm, word2idx, device, lang)
    mcqs = []
    with torch.no_grad():
        for sent, answer in zip(sents, answers):
            if not answer:  # Only skip if empty
                print(f"Skipping empty answer for '{sent}'", flush=True)
                continue
            sent_words = word_tokenize(sent)
            sent_ids = [word2idx.get(word, 0) for word in sent_words[:512]]
            if not sent_ids:
                sent_ids = [0]
            # *********************Akshay**********************
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

            # embedding = sentence_encoder.encode(sent, convert_to_tensor=False)
            # lstm_str = " ".join([f"{x:.4f}" for x in embedding[:10]])  
            # *********************Akshay**********************

            prompt = "generate question: " if lang == 'en' else "प्रश्न उत्पन्न करें: "
            input_text = f"{prompt}{sent} answer: {answer} [LSTM: {lstm_str}]"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
            question_ids = model.generate(**inputs, max_length=32, num_beams=4)
            question = tokenizer.decode(question_ids[0], skip_special_tokens=True)
            # if lang == 'hi' and not question.endswith(("क्या है?", "कहाँ है?", "कितने हैं?")):
            #     if "में" in sent and "स्थित" in sent:
            #         question += " कहाँ है?"
            #     elif "कितने" in sent:
            #         question += " कितने हैं?"
            #     else:
            #         question += " क्या है?"
            print(f"Generated question: {question}", flush=True)

            distractors = generate_distractors(answer, text, question, lang)
            options = [answer] + distractors
            random.shuffle(options)

            mcq = {
                "question": question,
                "options": options,
                "correct_answer": answer
            }
            mcqs.append(mcq)
            print(json.dumps(mcq, indent=2, ensure_ascii=False), flush=True)

    return mcqs

if __name__ == "__main__":
    # hindi_text = "भारत एक विशाल और विविधतापूर्ण देश है। नई दिल्ली भारत की राजधानी है और यहाँ कई ऐतिहासिक स्थल मौजूद हैं। मुंबई महाराष्ट्र की सबसे बड़ी नगरी है और इसे भारत की आर्थिक राजधानी भी कहा जाता है। ताजमहल आगरा में स्थित है और यह विश्व का एक प्रसिद्ध स्मारक है। गंगा नदी भारत की सबसे पवित्र नदी मानी जाती है और यह उत्तराखंड से निकलती है। कोलकाता पश्चिम बंगाल की राजधानी है और यह अपनी सांस्कृतिक विरासत के लिए जाना जाता है। भारत में 28 राज्य और 8 केंद्र शासित प्रदेश हैं। हिमालय पर्वत भारत के उत्तर में स्थित है और यह विश्व का सबसे ऊँचा पर्वत श्रृंखला है।"
    hindi_text = "एक आदर्श छात्र शिक्षा में अच्छा होता है, अतिरिक्त पाठ्यक्रम गतिविधियों में भाग लेता है, अच्छा व्यवहार करता है और सुंदर एवं सुशील दिखता है। हर व्यक्ति उसकी तरह बनना चाहता है और हर व्यक्ति उसका दोस्त बनना चाहता है। शिक्षक भी ऐसे छात्रों को पसंद करते हैं और वे जहां भी जाते हैं उनकी सराहना की जाती है। सभी इस तरह के छात्र के साथ बैठना या मित्रता करना चाहते हैं, तो कई लोग उनके लिए अच्छे की दुआ नहीं करते क्योंकि वे उनसे ईर्ष्या करते हैं। फिर भी इससे आदर्श छात्र के मन को कोई फर्क नहीं पड़ता क्योंकि वह अपने व्यक्तित्व से जीवन में उच्च चीजें हासिल करता है। आदर्श छात्र कोई ऐसा व्यक्ति नहीं है जो उत्तम है और प्रत्येक परीक्षा में पूर्ण अंक प्राप्त करता है या प्रत्येक खेल गतिविधियों, जिसमें वह भाग लेता है, और पदक जीतता है। आदर्श छात्र वह है जो अनुशासित रहता है और जीवन में एक सकारात्मक दृष्टि रखता है।"
    mcqs_hi = generate_mcq(hindi_text, 'hi')
    print(f"Generated {len(mcqs_hi)} MCQs for Hindi", flush=True)

    english_text = "Apache Hadoop software is an open source framework that allows for the distributed storage and processing of large datasets across clusters of computers using simple programming models. Hadoop is designed to scale up from a single computer to thousands of clustered computers, with each machine offering local computation and storage. In this way, Hadoop can efficiently store and process large datasets ranging in size from gigabytes to petabytes of data."
    mcqs_en = generate_mcq(english_text, 'en')
    print(f"Generated {len(mcqs_en)} MCQs for English", flush=True)
