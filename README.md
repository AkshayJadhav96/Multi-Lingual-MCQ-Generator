# Multilingual MCQ Generator

A hybrid LSTM + transformer + Adversarial Decoupling Module (ADM) approach to generate multiple-choice questions in English and Hindi. This project processes datasets like SQuAD (English) and IndicQA (Hindi), generates questions, and evaluates them using BLEU, METEOR, and ROUGE-L metrics.

![MCQ Generator](https://img.shields.io/badge/MCQ-Generator-blue)
![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Hindi-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Multilingual Support**: Generate MCQs in both English and Hindi
- **Hybrid Architecture**: Combines LSTM, transformer models, and Adversarial Decoupling Module
- **Comprehensive Evaluation**: Includes BLEU, METEOR, and ROUGE-L metrics
- **Optimized Hindi Pipeline**: Uses translation-based approach for improved Hindi MCQ quality

## Prerequisites

Before running the project, ensure the following are installed and configured:

- **Python 3.8+**: Required for all scripts
- **uv**: A Python package manager for managing dependencies and running scripts (below command is for linux)
  ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Pre-trained Embeddings**:
  - Download the following embeddings from [this Google Drive link](https://drive.google.com/drive/folders/1DeoZ8t5NPpEtB8LIXFI3BK8c12858XR8?usp=sharing):
    - `cc.hi.300.vec` (FastText embeddings for Hindi)
    - `glove.6B.300d.txt` (GloVe embeddings for English)
  - Instructions:
    1. Create the embeddings directory: `mkdir -p data/embeddings`
    2. Move the downloaded files to `data/embeddings/`

- **SpaCy and Stanza Models**:
  - English: `python -m spacy download en_core_web_sm`
  - Hindi: Will be downloaded automatically when running `data_preprocessing.py`

- **IndicNER Model**: Automatically downloaded when running `data_preprocessing.py` (uses `ai4bharat/indicner`)

## Installation

Follow these steps to set up and run the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AkshayJadhav96/Multi-Lingual-MCQ-Generator.git
   cd Multi-Lingual-MCQ-Generator
   uv sync
   ```

## Usage

All scripts are located in the `src/` directory. Use `uv` to run them:

1. **Data Preprocessing**:
   ```bash
   cd src
   uv run ak_data_preprocessing.py
   ```
   Output: `data/processed/english_processed.json`, `data/processed/hindi_processed.json`


2. **LSTM training**:
   ```bash
   cd src
   uv run ak_lstm_encoder.py
   ```
   Output: `models/lstm_english`, `models/lstm_hindi`


3. **Transformer Finetuning**:
   ```bash
   cd src
   uv run ak_transformer_finetune.py
   ```
   Output: `models/transformer/finetuned`

4. **Question Generation**: *(Note: This process is time-consuming when processing large datasets. For quicker MCQ generation on small texts, use options 6 or 7 below)*
   ```bash
   cd src
   uv run ak_question_generation.py
   ```
   Output: `data/generated/questions_en.json`, `data/generated/questions_hi.json`

5. **Evaluation**:
   ```bash
   cd src
   uv run evaluation.py
   ```
   Output: `data/evaluation/evaluation_en.json`, `data/evaluation/evaluation_hi.json`, `data/evaluation/evaluation_combined.json`


6. **Hindi MCQ Generation on custom text**:
   ```bash
   cd src
   uv run ak_generate_mcq.py
   ```
   This script will prompt you to enter or paste Hindi and English text for which you want to generate MCQs. It uses the base pipeline for MCQ generation.


7. **Improved Hindi MCQ Generation on custom text**:
   ```bash
   cd src
   uv run final.py
   ```
   This script will prompt you to enter or paste Hindi and English text uses an optimized pipeline (Hindi-to-English translation, English-based generation, back-translation) for better Hindi MCQ quality. Recommended for higher quality results.

## Model Architecture

The system uses a hybrid architecture:
- LSTM-based encoder for capturing sequential information
- Transformer components for contextual understanding
- Adversarial Decoupling Module (ADM) to separate language-specific and task-specific features

## Evaluation Metrics

The generated questions are evaluated using:
- **BLEU**: Measures n-gram precision
- **METEOR**: Evaluates semantic similarity
- **ROUGE-L**: Focuses on longest common subsequence

## Troubleshooting

- Ensure the `data/` directory contains the required datasets and embeddings before running scripts
- If you encounter issues with model paths (e.g., `lstm_english/model.pt`), verify that pre-trained models are in `models/`
- For Hindi MCQs, `final.py` is recommended as it leverages a translation-based approach to improve question quality
- Check console output and JSON files in `data/generated/` and `data/evaluation/` for results

## Acknowledgments

- SQuAD dataset for English question answering
- IndicQA dataset for Hindi question answering
- FastText and GloVe for pre-trained word embeddings