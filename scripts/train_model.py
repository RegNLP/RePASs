# scripts/train_model.py

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

def main():
    # Check if CUDA is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load and preprocess the data
    json_path = "./data/ObligationClassificationDataset.json"
    with open(json_path, 'r') as file:
        data = json.load(file)

    texts = [item['Text'] for item in data]
    labels = [1 if item['Obligation'] else 0 for item in data]  # Converting True/False to 1/0

    # Step 2: Tokenization using LegalBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

    class ObligationDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Splitting data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = ObligationDataset(X_train, y_train, tokenizer)
    val_dataset = ObligationDataset(X_val, y_val, tokenizer)

    # Step 3: Fine-tuning LegalBERT for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        'nlpaueb/legal-bert-base-uncased', num_labels=2
    )
    model.to(device)  # Move model to the GPU

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Step 4: Train the model
    trainer.train()

    # Step 5: Evaluate the model
    trainer.evaluate()

    # Step 6: Save the model and tokenizer for future use
    os.makedirs('./models/obligation-classifier-legalbert', exist_ok=True)
    model.save_pretrained('./models/obligation-classifier-legalbert')
    tokenizer.save_pretrained('./models/obligation-classifier-legalbert')

    print("Model fine-tuning and evaluation completed.")

if __name__ == "__main__":
    main()
