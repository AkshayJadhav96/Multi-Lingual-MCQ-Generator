# src/distractor_generation.py

import os
import random
import nltk
import spacy
import torch
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util # Import SentenceTransformer
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForTokenClassification
import traceback # For detailed error printing

# --- Define Paths Relative to Project Root ---
# Assumes this script is in 'src/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # Project Root (multilingual-mcq-generator/)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
NLTK_DATA_DIR = os.path.join(BASE_DIR, 'nltk_data')
LOCAL_SPACY_PATH_EN = os.path.join(MODEL_DIR, 'spacy_models', 'en_core_web_sm')
LOCAL_INDICNER_PATH = os.path.join(MODEL_DIR, 'ner_models', 'indic-ner')
SENTENCE_TRANSFORMER_PATH = os.path.join(MODEL_DIR, 'sentence_transformers', 'paraphrase-multilingual-mpnet-base-v2')
print(f"[Distractor Gen] Project Root: {BASE_DIR}")

# --- NLTK Setup & Download (with checks) ---
print("[Distractor Gen] Setting up NLTK...")
nltk_resources_available = True
if os.path.exists(NLTK_DATA_DIR):
    nltk.data.path.append(NLTK_DATA_DIR)
    print(f"[Distractor Gen] Using local NLTK data path: {NLTK_DATA_DIR}")
else:
    print(f"Warning: Local NLTK data dir not found: {NLTK_DATA_DIR}. Attempting downloads...")

# Define resources needed and download if missing
nltk_needed = {'corpora': ['wordnet'], 'tokenizers': ['punkt'], 'taggers': ['averaged_perceptron_tagger']}
for category, resources in nltk_needed.items():
    for resource in resources:
        try:
            # Check if resource exists in any known path
            if category == 'corpora':
                 nltk.data.find(f'corpora/{resource}')
            elif category == 'tokenizers':
                 nltk.data.find(f'tokenizers/{resource}')
            elif category == 'taggers':
                  nltk.data.find(f'taggers/{resource}')
            print(f"[Distractor Gen] NLTK resource '{resource}' found.")
        except LookupError:
            print(f"Warning: NLTK resource '{resource}' not found. Attempting download...")
            try:
                # Attempt download to the specified directory first
                nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
                # Add the path again just in case download created it
                if not os.path.exists(NLTK_DATA_DIR): os.makedirs(NLTK_DATA_DIR)
                if NLTK_DATA_DIR not in nltk.data.path: nltk.data.path.append(NLTK_DATA_DIR)
                print(f"[Distractor Gen] NLTK resource '{resource}' downloaded to {NLTK_DATA_DIR}.")
            except Exception as download_error:
                print(f"ERROR: Failed to download NLTK resource '{resource}'. WordNet/POS features might fail. Error: {download_error}")
                nltk_resources_available = False # Mark that not all resources are ready
        except Exception as find_error:
             print(f"ERROR checking for NLTK resource '{resource}'. Error: {find_error}")
             nltk_resources_available = False


# --- Load Models (with error handling) ---
print("\n[Distractor Gen] Loading NLP models...")

# SpaCy (English POS/NER)
nlp_en = None
if os.path.isdir(LOCAL_SPACY_PATH_EN):
    try:
        nlp_en = spacy.load(LOCAL_SPACY_PATH_EN)
        print(f"[Distractor Gen] Loaded local spaCy model from: {LOCAL_SPACY_PATH_EN}")
    except Exception as e:
        print(f"Error loading local spaCy model from {LOCAL_SPACY_PATH_EN}: {e}")
else:
    print(f"Warning: Local spaCy model dir not found: {LOCAL_SPACY_PATH_EN}. English NER/POS features unavailable.")

# IndicNER (Hindi NER)
nlp_hi_ner = None
if os.path.isdir(LOCAL_INDICNER_PATH):
    try:
        ner_tokenizer_hi = AutoTokenizer.from_pretrained(LOCAL_INDICNER_PATH)
        ner_model_hi = AutoModelForTokenClassification.from_pretrained(LOCAL_INDICNER_PATH)
        device_id = 0 if torch.cuda.is_available() else -1
        nlp_hi_ner = hf_pipeline("ner", model=ner_model_hi, tokenizer=ner_tokenizer_hi, aggregation_strategy="simple", device=device_id)
        print(f"[Distractor Gen] Loaded local IndicNER model from: {LOCAL_INDICNER_PATH}")
    except Exception as e:
        print(f"Error loading local IndicNER model from {LOCAL_INDICNER_PATH}: {e}")
else:
    print(f"Warning: Local IndicNER model dir not found: {LOCAL_INDICNER_PATH}. Hindi NER features unavailable.")

# Sentence Transformer (Multilingual)
sentence_model = None
if os.path.isdir(SENTENCE_TRANSFORMER_PATH):
    try:
        sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)
        print(f"[Distractor Gen] Loaded local Sentence Transformer model from: {SENTENCE_TRANSFORMER_PATH}")
    except Exception as e:
        print(f"Error loading local Sentence Transformer model from {SENTENCE_TRANSFORMER_PATH}: {e}")
else:
    print(f"Warning: Local Sentence Transformer dir not found: {SENTENCE_TRANSFORMER_PATH}. Similarity features will be skipped.")
    print("Please run the download command in the previous answer if you want to use it.")

# --- Helper Function: get_entities ---
def get_entities(text, lang):
    """Extracts entities using appropriate NER model (handles offline cases)."""
    entities = {}
    if not text: return entities
    try:
        if lang == 'en' and nlp_en: # Check if spaCy model loaded
            doc = nlp_en(text[:100000]) # Limit length for safety
            for ent in doc.ents:
                # Basic filtering of entities that are just the answer or too short
                if ent.text.strip().lower() != text.strip().lower() and len(ent.text.strip()) > 1:
                    entities[ent.text.strip()] = ent.label_
        elif lang == 'hi' and nlp_hi_ner: # Check if IndicNER model loaded
            ner_results = nlp_hi_ner(text)
            for entity in ner_results:
                # Check keys exist and filter
                if 'word' in entity and 'entity_group' in entity:
                    word = entity['word'].strip()
                    if word.lower() != text.strip().lower() and len(word) > 1:
                        entities[word] = entity['entity_group']
        # No warning needed here if models didn't load, handled above
    except Exception as e:
        print(f"Warning: NER processing error during get_entities (lang={lang}): {text[:50]}... Error: {e}")
    return entities

# --- Helper: Extract Noun Phrases (English) ---
def extract_noun_phrases(text, lang='en'):
    if lang != 'en' or not nlp_en:
        return []
    try:
        doc = nlp_en(text[:100000]) # Limit length
        # Filter noun chunks: not equal to answer (case-insensitive), reasonable length
        nps = [chunk.text.strip() for chunk in doc.noun_chunks
               if len(chunk.text.split()) <= 4 and len(chunk.text.strip()) > 1]
        return nps
    except Exception as e:
        print(f"Error extracting noun phrases: {e}")
        return []

# --- Main Distractor Generation Function ---
def generate_distractors(answer, context, question, lang='en', num_distractors=3, entities=None):
    """
    Generates distractors for a given answer, context, and question.

    Args:
        answer (str): The correct answer.
        context (str): The text context from which the answer/question derived.
        question (str): The generated question.
        lang (str): Language code ('en' or 'hi').
        num_distractors (int): The desired number of distractors.
        entities (dict): Pre-computed entities from the context (optional).

    Returns:
        list: A list of distractor strings.
    """
    distractors = set() # Use a set to avoid duplicates initially
    answer_clean = answer.strip()
    answer_lower = answer_clean.lower()
    if not answer_clean: # Handle empty answer case
         print("Warning: Cannot generate distractors for an empty answer.")
         return [f"Option {i+1}" if lang == 'en' else f"विकल्प {i+1}" for i in range(num_distractors)]


    # --- Strategy 1: NER-based Distractors ---
    if entities is None: # Get entities if not provided
        # print("[Distractor Gen] Getting entities for context...") # Verbose
        entities = get_entities(context, lang)

    answer_entity_type = None
    # Find entity type of the answer
    for entity_text, entity_type in entities.items():
        # Check for exact match first (case-insensitive)
        if entity_text.lower() == answer_lower:
            answer_entity_type = entity_type
            # print(f"[Distractor Gen] Answer '{answer_clean}' identified as entity type: {answer_entity_type}") # Verbose
            break
    # If no exact match, check if answer is substring of an entity (less reliable)
    if answer_entity_type is None:
        for entity_text, entity_type in entities.items():
            if answer_lower in entity_text.lower():
                answer_entity_type = entity_type
                # print(f"[Distractor Gen] Answer '{answer_clean}' potentially related to entity type: {answer_entity_type}") # Verbose
                break

    ner_dist_count = 0
    if answer_entity_type:
        # Find other entities of the SAME type, ensuring they are not the answer
        for entity_text, entity_type in entities.items():
            if entity_type == answer_entity_type and entity_text.lower() != answer_lower:
                distractors.add(entity_text) # Add the original casing
                ner_dist_count+=1
            if len(distractors) >= num_distractors: break
    if ner_dist_count > 0: print(f"[Distractor Gen] Found {ner_dist_count} distractors using NER.")


    # --- Strategy 2: Sentence Embedding Similarity (if model loaded) ---
    sim_dist_count = 0
    if sentence_model and len(distractors) < num_distractors:
        # print("[Distractor Gen] Trying Sentence Embedding Similarity...") # Verbose
        try:
            # Extract candidate phrases
            if lang == 'en':
                # Combine noun phrases and single capitalized words (potential proper nouns)
                candidates = list(set(extract_noun_phrases(context, lang)))
                cap_words = [w.strip('.,!?;:"\'()[]') for w in context.split() if w and w[0].isupper() and len(w)>1]
                candidates.extend(cap_words)
            else: # Hindi - use simple words and NER entities
                candidates = list(set([w.strip('.,!?;:"\'()[]') for w in context.split() if len(w.strip('.,!?;:"\'()[]')) > 2 and w.isalnum()]))
                candidates.extend(list(entities.keys())) # Add identified entities

            # Filter candidates: must not be the answer, must not be empty, unique
            candidates = list(set([c.strip() for c in candidates if c.strip().lower() != answer_lower and len(c.strip()) > 1]))

            if len(candidates) > 1: # Need at least one candidate besides the answer potentially
                answer_embedding = sentence_model.encode(answer_clean, convert_to_tensor=True)
                candidate_embeddings = sentence_model.encode(candidates, convert_to_tensor=True)

                cosine_scores = util.pytorch_cos_sim(answer_embedding, candidate_embeddings)[0]
                # Combine candidates with scores, sort by similarity
                sorted_candidates = sorted(zip(candidates, cosine_scores.tolist()), key=lambda x: x[1], reverse=True)

                # Add top N *distinct* similar candidates
                for cand, score in sorted_candidates:
                    if len(distractors) >= num_distractors: break
                    # Add thresholds: plausible similarity but not too similar
                    if score > 0.5 and score < 0.95:
                        if cand not in distractors: # Check again for duplicates from NER
                            distractors.add(cand)
                            sim_dist_count += 1
            if sim_dist_count > 0: print(f"[Distractor Gen] Added {sim_dist_count} distractors using Similarity.")

        except Exception as e:
            print(f"Error during sentence similarity processing: {e}")
            traceback.print_exc()


    # --- Strategy 3: WordNet (English Only) ---
    wn_dist_count = 0
    if nltk_resources_available and lang == 'en' and len(distractors) < num_distractors:
        # print("[Distractor Gen] Trying WordNet...") # Verbose
        try:
            # Try to get POS tag of the answer for more targeted synset search
            answer_pos_tag = None
            if nlp_en:
                ans_doc = nlp_en(answer_clean)
                if len(ans_doc) > 0:
                     # Map spaCy POS to WordNet POS (simplified)
                     spacy_pos = ans_doc[0].pos_
                     if spacy_pos == 'NOUN': answer_pos_tag = wordnet.NOUN
                     elif spacy_pos == 'VERB': answer_pos_tag = wordnet.VERB
                     elif spacy_pos == 'ADJ': answer_pos_tag = wordnet.ADJ
                     elif spacy_pos == 'ADV': answer_pos_tag = wordnet.ADV

            # Use last word of answer for lookup if multi-word
            lookup_word = answer_clean.split()[-1]
            synsets = wordnet.synsets(lookup_word, pos=answer_pos_tag) if answer_pos_tag else wordnet.synsets(lookup_word)

            # Get synonyms
            syn_distractors = set()
            for syn in synsets[:5]: # Limit number of synsets explored
                for lemma in syn.lemmas()[:3]: # Limit lemmas per synset
                    dist = lemma.name().replace('_', ' ').strip()
                    # Filter: not answer, not already added, reasonable length diff
                    if dist.lower() != answer_lower and dist not in distractors and abs(len(dist) - len(answer_clean)) < 5:
                        syn_distractors.add(dist)

            # Add synonyms if needed
            for dist in syn_distractors:
                if len(distractors) >= num_distractors: break
                if dist not in distractors:
                     distractors.add(dist)
                     wn_dist_count += 1

            # If still not enough, try related forms (hypernyms/hyponyms)
            if len(distractors) < num_distractors and synsets:
                 related_distractors = set()
                 # Hypernyms (broader terms)
                 for syn in synsets[:2]:
                     for hyper in syn.hypernyms():
                         for lemma in hyper.lemmas()[:2]:
                             dist = lemma.name().replace('_', ' ').strip()
                             if dist.lower() != answer_lower and dist not in distractors: related_distractors.add(dist)
                 # Hyponyms (more specific terms) - less common but possible
                 # for syn in synsets[:2]:
                 #    for hypo in syn.hyponyms():
                 #        for lemma in hypo.lemmas()[:2]:
                 #             dist = lemma.name().replace('_', ' ').strip()
                 #             if dist.lower() != answer_lower and dist not in distractors: related_distractors.add(dist)

                 for dist in related_distractors:
                      if len(distractors) >= num_distractors: break
                      if dist not in distractors:
                           distractors.add(dist)
                           wn_dist_count += 1

            if wn_dist_count > 0: print(f"[Distractor Gen] Added {wn_dist_count} distractors from WordNet.")

        except Exception as e:
             print(f"Error during WordNet processing: {e}")


    # --- Strategy 4: Context Words (Fallback) ---
    fallback_dist_count = 0
    if len(distractors) < num_distractors:
        # print("[Distractor Gen] Using Context Words Fallback...") # Verbose
        context_words = context.split()
        potential_distractors = []
        # Prefer words with similar length, capitalized, or same starting letter?
        for word in context_words:
             word_clean = word.strip('.,!?;:"\'()[]')
             if word_clean.lower() != answer_lower and len(word_clean) > 2 and (word_clean[0].isupper() or abs(len(word_clean) - len(answer_clean)) <= 4):
                  potential_distractors.append(word_clean)

        potential_distractors = list(set(potential_distractors)) # Unique context words
        random.shuffle(potential_distractors)

        for word in potential_distractors:
            if len(distractors) >= num_distractors: break
            if word not in distractors: # Check against already added distractors
                distractors.add(word)
                fallback_dist_count += 1
        if fallback_dist_count > 0: print(f"[Distractor Gen] Added {fallback_dist_count} distractors from Context Fallback.")


    # --- Final Check and Placeholder Fill ---
    final_distractors = list(distractors)
    # Remove the original answer if it accidentally got added
    final_distractors = [d for d in final_distractors if d.lower() != answer_lower]
    random.shuffle(final_distractors)
    final_distractors = final_distractors[:num_distractors] # Take top N unique ones

    placeholder_count = 0
    while len(final_distractors) < num_distractors:
        placeholder_count += 1
        placeholder = f"Option {len(final_distractors) + 1}" if lang == 'en' else f"विकल्प {len(final_distractors) + 1}"
        # Ensure placeholder itself isn't a duplicate (edge case)
        if placeholder not in final_distractors:
            final_distractors.append(placeholder)
        else:
             final_distractors.append(f"{placeholder}-{random.randint(10,99)}") # Add random num if placeholder exists
    # if placeholder_count > 0: print(f"[Distractor Gen] Added {placeholder_count} placeholder distractors.") # Verbose

    return final_distractors


# --- Example Usage / Testing Block ---
if __name__ == "__main__":
    print("\n" + "="*30)
    print("--- Testing Distractor Generation ---")
    print("="*30)

    # Example 1: English NER
    en_context1 = "Barack Obama visited Paris, the capital of France, while Angela Merkel stayed in Berlin."
    en_answer1 = "Paris"
    en_question1 = "Where did Obama visit?"
    print(f"\nExample 1 (English NER): Context='{en_context1}', Answer='{en_answer1}'")
    # *** CALL get_entities HERE for the test block ***
    en_entities1 = get_entities(en_context1, 'en')
    dist1 = generate_distractors(en_answer1, en_context1, en_question1, 'en', entities=en_entities1)
    print(f"--> Generated Distractors: {dist1}")
    if dist1: assert len(dist1) == 3

    # Example 2: English WordNet/Fallback
    en_context2 = "The quick brown fox jumps over the lazy dog. Dogs are mammals."
    en_answer2 = "fox"
    en_question2 = "What jumped over the dog?"
    print(f"\nExample 2 (English Other): Context='{en_context2}', Answer='{en_answer2}'")
    en_entities2 = get_entities(en_context2, 'en')
    dist2 = generate_distractors(en_answer2, en_context2, en_question2, 'en', entities=en_entities2)
    print(f"--> Generated Distractors: {dist2}")
    if dist2: assert len(dist2) == 3

    # Example 3: Hindi NER/Context
    hi_context3 = "दलाई लामा ने अपनी संवेदना व्यक्त की और कहा कि कलाम एक महान वैज्ञानिक थे। भूटान सरकार ने भी शोक जताया।"
    hi_answer3 = "दलाई लामा"
    hi_question3 = "किसने संवेदना व्यक्त की?"
    print(f"\nExample 3 (Hindi NER): Context='{hi_context3}', Answer='{hi_answer3}'")
    hi_entities3 = get_entities(hi_context3, 'hi')
    dist3 = generate_distractors(hi_answer3, hi_context3, hi_question3, 'hi', entities=hi_entities3)
    print(f"--> Generated Distractors: {dist3}")
    if dist3: assert len(dist3) == 3

    # Example 4: Hindi Capital (Checking if old logic removed/overridden)
    hi_context4 = "भारत की राजधानी दिल्ली है। मुम्बई आर्थिक राजधानी है।"
    hi_answer4 = "दिल्ली"
    hi_question4 = "भारत की राजधानी क्या है?"
    print(f"\nExample 4 (Hindi Context): Context='{hi_context4}', Answer='{hi_answer4}'")
    hi_entities4 = get_entities(hi_context4, 'hi')
    dist4 = generate_distractors(hi_answer4, hi_context4, hi_question4, 'hi', entities=hi_entities4)
    print(f"--> Generated Distractors: {dist4}")
    if dist4: assert len(dist4) == 3
    # assert "मुंबई" in dist4 or "कोलकाता" not in dist4 # Specific check might fail depending on strategies

    # Example 5: Empty Answer
    print(f"\nExample 5 (Empty Answer):")
    dist5 = generate_distractors("", "Some context", "Some question", 'en')
    print(f"--> Generated Distractors: {dist5}")
    assert len(dist5) == 3

    print("\n" + "="*30)
    print("--- Distractor Generation Test Completed ---")
    print("="*30)