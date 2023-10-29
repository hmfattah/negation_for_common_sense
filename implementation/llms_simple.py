import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import RobertaTokenizer
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import EarlyStoppingCallback

from datasets import Dataset, DatasetDict
from datasets import load_dataset
from datasets import load_metric

from sklearn.metrics import classification_report
from scipy.special import softmax
import pandas as pd
import numpy as np
import json
import csv
import random

#pip install datasets
#pip install transformers
#pip install scikit-learn
#pip install scipy
#pip install torch
#pip install accelerate==0.20.3

root="../data"
link_test_all = root+'/2_test_data_1488.csv'
link4 = root+'/1_atomic_all_p.csv'

test_data_all = pd.read_csv(link_test_all)
atomic_data = pd.read_csv(link4)

#concat all and make a new string
def concat_all_by_sep_train(example):
  output = int(example['output'])
  final_str = example['p'] + " </s> " + example['r'] + " </s> " + example['q']
  return {'label': output, 'text': final_str}

atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
train_data = atomic_data.head(1000) 

td = Dataset.from_pandas(train_data)
if '__index_level_0__' in td.column_names:
    td = td.remove_columns(['__index_level_0__'])

# Filter out rows where 'q' column has value 'nan'
filtered_dataset = td.filter(lambda example: example['q'] != 'nan')
filtered_dataset = filtered_dataset.filter(lambda example: example['q'] is not None)
train_dataset = filtered_dataset.map(concat_all_by_sep_train)

new_train_dataset = train_dataset.remove_columns(['p', 'q', 'r', 'output'])
new_train_dataset = new_train_dataset.shuffle(seed=42)

test_dataset_all = Dataset.from_pandas(test_data_all)
test_dataset_all = test_dataset_all.map(concat_all_by_sep_train)
test_dataset_all

new_test_dataset_2 = test_dataset_all
if '__index_level_0__' in test_dataset_all.column_names:
    new_test_dataset_2 = test_dataset_all.remove_columns(['__index_level_0__'])
new_test_dataset_2
new_test_dataset_2 = new_test_dataset_2.remove_columns(['p', 'q', 'r', 'output'])
print(new_test_dataset_2[0])
print(new_test_dataset_2[1])

dts = new_train_dataset.train_test_split(test_size=0.10)
  
dataset = DatasetDict()
dataset['train'] = Dataset.from_pandas(dts["train"].to_pandas())
dataset['validation'] = Dataset.from_pandas(dts["test"].to_pandas())
dataset['test'] =  Dataset.from_pandas(new_test_dataset_2.to_pandas())

print(dataset)

# Define your training, evaluation, and test data
# Assuming you have datasets with 'text' and 'label' columns where 'label' is 0 or 1
# You'll need to load your data into these datasets or dataframes

training_data = dataset['train']
evaluation_data = dataset['validation']
test_data = dataset['test']

# Load the BART-Large model and tokenizer
checkpoint = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Define the number of labels (in this case, 2 for binary classification)
num_labels = 2

# Modify the BART-Large model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

# Tokenize your training and evaluation data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_training_data = training_data.map(tokenize_function, batched=True)
tokenized_evaluation_data = evaluation_data.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Adjust the number of training epochs
    evaluation_strategy="steps",
    save_total_limit=2,
    
    learning_rate=1e-4,  # Adjust the learning rate
    
    per_device_train_batch_size=8,  # Adjust batch size
    per_device_eval_batch_size=8,  # Adjust batch size
    
    logging_dir="./logs",
    #evaluation_steps=500,  # Evaluate every 500 steps
    #save_steps=500,  # Save every 500 steps
    save_strategy="steps",
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
)

# Define a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_data,
    eval_dataset=tokenized_evaluation_data,
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
tokenized_test_data = test_data.map(tokenize_function, batched=True)
results = trainer.evaluate(eval_dataset=tokenized_test_data)

print("Results:", results)