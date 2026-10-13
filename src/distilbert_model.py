import os
import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

def tokenize_datasets(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
    """Loading tokenizer for DistilBERT and tokenizing tran, val and test data"""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenize_val = val_dataset.map(tokenize_function, batched=True)
    tokenize_test = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train, tokenize_val, tokenize_test, tokenizer

def get_model_and_collator(tokenizer: AutoTokenizer):
    """Loading DistilBERT model for classification and creating a data collator"""
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return model, data_collator

def setup_trainer(model, tokenized_train, tokenized_val, data_collator) -> Trainer:
    """Setting up training arguments, metrics and creating Trainer object"""
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./tmp_trainer",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    return trainer