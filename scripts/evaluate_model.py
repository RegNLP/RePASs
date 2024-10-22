import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize as sent_tokenize_uncached
import nltk
from functools import cache
import tqdm
import os
import csv

nltk.download('punkt')

# Set up device for torch operations
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Correct path to the trained model and tokenizer
model_name = './models/obligation-classifier-legalbert'

# Load the tokenizer and model for obligation detection
obligation_tokenizer = AutoTokenizer.from_pretrained(model_name)
obligation_model = AutoModelForSequenceClassification.from_pretrained(model_name)
obligation_model.to(device)
obligation_model.eval()

# Load NLI model and tokenizer for obligation coverage using Microsoft's model
coverage_nli_model = pipeline("text-classification", model="microsoft/deberta-large-mnli", device=device)

# Load NLI model and tokenizer for entailment and contradiction checks
nli_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-xsmall')
nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-xsmall')
nli_model.to(device)
nli_model.eval()

# Define a cached version of sentence tokenization
@cache
def sent_tokenize(passage: str):
    return sent_tokenize_uncached(passage)

def softmax(logits):
    e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_logits / np.sum(e_logits, axis=1, keepdims=True)

def get_nli_probabilities(premises, hypotheses):
    features = nli_tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt").to(device)
    nli_model.eval()
    with torch.no_grad():
        logits = nli_model(**features).logits.cpu().numpy()
    probabilities = softmax(logits)
    return probabilities

def get_nli_matrix(passages, answers):
    entailment_matrix = np.zeros((len(passages), len(answers)))
    contradiction_matrix = np.zeros((len(passages), len(answers)))

    batch_size = 16
    for i, pas in enumerate(tqdm.tqdm(passages)):
        for b in range(0, len(answers), batch_size):
            e = b + batch_size
            probs = get_nli_probabilities([pas] * len(answers[b:e]), answers[b:e])  # Get NLI probabilities
            entailment_matrix[i, b:e] = probs[:, 1]
            contradiction_matrix[i, b:e] = probs[:, 0]
    return entailment_matrix, contradiction_matrix

def calculate_scores_from_matrix(nli_matrix, score_type='entailment'):
    if nli_matrix.size == 0:
        return 0.0
    return np.round(np.mean(np.max(nli_matrix, axis=0)), 5)

def classify_obligations(sentences):
    inputs = obligation_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = obligation_model(**inputs).logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return predictions

def calculate_obligation_coverage_score(passages, answers):
    # Filter obligation sentences from passages
    obligation_sentences_source = []
    for passage in passages:
        sentences = sent_tokenize(passage)
        is_obligation = classify_obligations(sentences)
        obligation_sentences_source.extend([sent for sent, label in zip(sentences, is_obligation) if label == 1])

    # Filter obligation sentences from answers
    obligation_sentences_answer = []
    for answer in answers:
        sentences = sent_tokenize(answer)
        is_obligation = classify_obligations(sentences)
        obligation_sentences_answer.extend([sent for sent, label in zip(sentences, is_obligation) if label == 1])

    # Calculate coverage based on NLI entailment
    covered_count = 0
    for obligation in obligation_sentences_source:
        for answer_sentence in obligation_sentences_answer:
            nli_result = coverage_nli_model(f"{answer_sentence} [SEP] {obligation}")
            if nli_result[0]['label'].lower() == 'entailment' and nli_result[0]['score'] > 0.7:
                covered_count += 1
                break

    return covered_count / len(obligation_sentences_source) if obligation_sentences_source else 0


def calculate_final_composite_score(passages, answers):
    passage_sentences = [sent for passage in passages for sent in sent_tokenize(passage)]
    answer_sentences = [sent for answer in answers for sent in sent_tokenize(answer)]
    entailment_matrix, contradiction_matrix = get_nli_matrix(passage_sentences, answer_sentences)
    entailment_score = calculate_scores_from_matrix(entailment_matrix, 'entailment')
    contradiction_score = calculate_scores_from_matrix(contradiction_matrix, 'contradiction')
    obligation_coverage_score = calculate_obligation_coverage_score(passages, answers)

    composite_score = (obligation_coverage_score + entailment_score - contradiction_score + 1) / 3
    return np.round(composite_score, 5), entailment_score, contradiction_score, obligation_coverage_score

def main(input_file_path, group_method_name):
    # Create a directory with the group_method_name in the data folder
    output_dir = f'./data/{group_method_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Define the paths for result files
    output_file_csv = os.path.join(output_dir, 'results.csv')
    output_file_txt = os.path.join(output_dir, 'results.txt')

    processed_question_ids = set()
    composite_scores = []
    entailment_scores = []
    contradiction_scores = []
    obligation_coverage_scores = []

    # Check if the output CSV file already exists and read processed QuestionIDs
    if os.path.exists(output_file_csv):
        with open(output_file_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_question_ids.add(row['QuestionID'])

    with open(input_file_path, 'r') as file:
        test_data = json.load(file)

    # Open the CSV file for appending results
    with open(output_file_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not processed_question_ids:
            # Write the header if the file is empty or new
            writer.writerow(['QuestionID', 'entailment_score', 'contradiction_score', 'obligation_coverage_score', 'composite_score'])

        for item in tqdm.tqdm(test_data):
            question_id = item['QuestionID']

            # Skip if the QuestionID has already been processed
            if question_id in processed_question_ids:
                print(f"Skipping QuestionID {question_id}, already processed.")
                continue

            # Skip if the "Answer" is null or empty
            if not item.get('Answer') or not item['Answer'].strip():
                print(f"Skipping QuestionID {question_id}, no answer.")
                continue

            # Merge "RetrievedPassages" if it's a list
            if isinstance(item['RetrievedPassages'], list):
                item['RetrievedPassages'] = " ".join(item['RetrievedPassages'])

            passages = [item['RetrievedPassages']]
            answers = [item['Answer']]
            composite_score, entailment_score, contradiction_score, obligation_coverage_score = calculate_final_composite_score(passages, answers)

            # Append the scores to the lists
            composite_scores.append(composite_score)
            entailment_scores.append(entailment_score)
            contradiction_scores.append(contradiction_score)
            obligation_coverage_scores.append(obligation_coverage_score)

            # Write the result to the CSV file
            writer.writerow([question_id, entailment_score, contradiction_score, obligation_coverage_score, composite_score])

    # Calculate averages
    avg_entailment = np.mean(entailment_scores)
    avg_contradiction = np.mean(contradiction_scores)
    avg_obligation_coverage = np.mean(obligation_coverage_scores)
    avg_composite = np.mean(composite_scores)

    # Print and save results to a text file
    results = (
        f"Average Entailment Score: {avg_entailment}\n"
        f"Average Contradiction Score: {avg_contradiction}\n"
        f"Average Obligation Coverage Score: {avg_obligation_coverage}\n"
        f"Average Final Composite Score: {avg_composite}\n"
    )

    print(results)

    with open(output_file_txt, 'w') as txtfile:
        txtfile.write(results)

    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Obligation Coverage")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--group_method_name', type=str, required=True, help='Method name for grouping results')

    args = parser.parse_args()

    main(args.input_file, args.group_method_name)
