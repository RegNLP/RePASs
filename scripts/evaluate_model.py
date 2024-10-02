# scripts/evaluate_model.py

import json
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from nltk.tokenize import sent_tokenize as sent_tokenize_uncached
import nltk
from functools import cache
import tqdm
import os

def setup():
    nltk.download('punkt')

@cache
def sent_tokenize(passage: str):
    return sent_tokenize_uncached(passage)

def softmax(logits):
    e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_logits / np.sum(e_logits, axis=1, keepdims=True)

def get_nli_probabilities(tokenizer, nli_model, premises, hypotheses, device):
    features = tokenizer(
        premises, hypotheses, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    nli_model.eval()
    with torch.no_grad():
        logits = nli_model(**features).logits.cpu().numpy()
    probabilities = softmax(logits)
    return probabilities

def get_nli_matrix(tokenizer, nli_model, passages, answers, device):
    print(f"{len(passages)} passages and {len(answers)} answers.")
    entailment_matrix = np.zeros((len(passages), len(answers)))
    contradiction_matrix = np.zeros((len(passages), len(answers)))

    batch_size = 16
    for i, pas in enumerate(tqdm.tqdm(passages)):
        for b in range(0, len(answers), batch_size):
            e = b + batch_size
            probs = get_nli_probabilities(
                tokenizer, nli_model, [pas] * (e - b), answers[b:e], device
            )  # Get NLI probabilities
            entailment_matrix[i, b:e] = probs[:, 1]
            contradiction_matrix[i, b:e] = probs[:, 0]
    return entailment_matrix, contradiction_matrix

def calculate_scores_from_matrix(nli_matrix, score_type='entailment'):
    if nli_matrix.size == 0:
        print("Warning: NLI matrix is empty. Returning default score of 0.")
        return 0.0  # or some other default score or handling as appropriate for your use case

    if score_type == 'entailment':
        reduced_vector = np.max(nli_matrix, axis=0)
    elif score_type == 'contradiction':
        reduced_vector = np.max(nli_matrix, axis=0)
    score = np.round(np.mean(reduced_vector), 5)
    return score

def calculate_obligation_coverage_score(coverage_pipeline, passages, answers):
    obligation_sentences_source = [sent for passage in passages for sent in sent_tokenize(passage)]
    obligation_sentences_answer = [sent for answer in answers for sent in sent_tokenize(answer)]
    covered_count = 0

    for obligation in obligation_sentences_source:
        obligation_covered = False
        for answer_sentence in obligation_sentences_answer:
            nli_result = coverage_pipeline(f"{answer_sentence} [SEP] {obligation}")
            if nli_result[0]['label'].lower() == 'entailment' and nli_result[0]['score'] > 0.7:
                covered_count += 1
                obligation_covered = True
                break

    coverage_score = covered_count / len(obligation_sentences_source) if obligation_sentences_source else 0
    return coverage_score

def calculate_final_composite_score(passages, answers, tokenizer, nli_model, coverage_pipeline, device):
    passage_sentences = [sent for passage in passages for sent in sent_tokenize(passage)]
    answer_sentences = [sent for answer in answers for sent in sent_tokenize(answer)]
    entailment_matrix, contradiction_matrix = get_nli_matrix(
        tokenizer, nli_model, passage_sentences, answer_sentences, device
    )
    entailment_score = calculate_scores_from_matrix(entailment_matrix, 'entailment')
    contradiction_score = calculate_scores_from_matrix(contradiction_matrix, 'contradiction')
    obligation_coverage_score = calculate_obligation_coverage_score(
        coverage_pipeline, passages, answers
    )
    print(f"Entailment Score: {entailment_score}")
    print(f"Contradiction Score: {contradiction_score}")
    print(f"Obligation Coverage Score: {obligation_coverage_score}")

    # New formula: (O + E - C + 1) / 3
    composite_score = (obligation_coverage_score + entailment_score - contradiction_score + 1) / 3
    print(f"Final Composite Score: {composite_score}")
    return np.round(composite_score, 5)

def main(input_file_path, group_methodName):
    setup()

    # Set up device for torch operations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer and model for obligation detection
    model_name = './models/obligation-classifier-legalbert'
    obligation_tokenizer = AutoTokenizer.from_pretrained(model_name)
    obligation_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    obligation_model.to(device)
    obligation_model.eval()

    # Load NLI model and tokenizer for obligation coverage using Microsoft's model
    coverage_nli_model = pipeline(
        "text-classification", model="microsoft/deberta-large-mnli", device=0 if torch.cuda.is_available() else -1
    )

    # Load NLI model and tokenizer for entailment and contradiction checks
    nli_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-xsmall')
    nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-xsmall')
    nli_model.to(device)
    nli_model.eval()

    with open(input_file_path, 'r') as file:
        test_data = json.load(file)

    # Prepare the data
    composite_scores = []
    entailment_scores = []
    contradiction_scores = []
    obligation_coverage_scores = []
    total_items = len(test_data)

    for index, item in enumerate(test_data):

        # Merge "RetrievedPassages" if it's a list
        if isinstance(item['RetrievedPassages'], list):
            item['RetrievedPassages'] = " ".join(item['RetrievedPassages'])

        question = [item['QuestionID']]
        passages = [item['RetrievedPassages']]
        answers = [item['Answer']]
        print(f"Processing {index + 1}/{total_items}: QuestionID {question}")

        # Calculate and store scores
        passage_sentences = [sent for passage in passages for sent in sent_tokenize(passage)]
        answer_sentences = [sent for answer in answers for sent in sent_tokenize(answer)]
        entailment_matrix, contradiction_matrix = get_nli_matrix(
            nli_tokenizer, nli_model, passage_sentences, answer_sentences, device
        )
        entailment_score = calculate_scores_from_matrix(entailment_matrix, 'entailment')
        contradiction_score = calculate_scores_from_matrix(contradiction_matrix, 'contradiction')
        obligation_coverage_score = calculate_obligation_coverage_score(
            coverage_nli_model, passages, answers
        )
        final_composite_score = (
            obligation_coverage_score + entailment_score - contradiction_score + 1
        ) / 3

        # Append to respective lists
        entailment_scores.append(entailment_score)
        contradiction_scores.append(contradiction_score)
        obligation_coverage_scores.append(obligation_coverage_score)
        composite_scores.append(final_composite_score)

    # Calculate averages
    avg_entailment = np.mean(entailment_scores)
    avg_contradiction = np.mean(contradiction_scores)
    avg_obligation_coverage = np.mean(obligation_coverage_scores)
    avg_composite = np.mean(composite_scores)

    # Print and save results to a file
    results = (
        f"Average Entailment Score: {avg_entailment}\n"
        f"Average Contradiction Score: {avg_contradiction}\n"
        f"Average Obligation Coverage Score: {avg_obligation_coverage}\n"
        f"Average Final Composite Score: {avg_composite}\n"
    )
    print(group_methodName)
    print(results)

    # Save the results to a text file
    output_file_path = f"{group_methodName}Results.txt"
    with open(output_file_path, 'w') as output_file:
        output_file.write(results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Obligation Coverage")
    parser.add_argument(
        '--input_file', type=str, required=True, help='Path to the input JSON file'
    )
    parser.add_argument(
        '--group_method_name', type=str, required=True, help='Method name for grouping results'
    )

    args = parser.parse_args()

    main(args.input_file, args.group_method_name)
