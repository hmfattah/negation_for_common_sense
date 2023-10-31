#pip install datasets
#pip install transformers
#pip install scikit-learn
#pip install scipy
#pip install torch
#pip install accelerate==0.20.3

from datasets import Dataset, DatasetDict
import pandas as pd
from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import TrainingArguments, Trainer
from scipy.special import softmax
from sklearn.metrics import classification_report
from transformers import EarlyStoppingCallback
import json
import csv
import random
import os 
from sklearn.dummy import DummyClassifier
from transformers import AutoTokenizer

import torch
#torch.cuda.empty_cache()

root="../data"
link2 = root+'/2_test_data.csv'
link_test_all = root+'/2_test_data_1488.csv'
link3 = root+'/1_annotated_data.csv'

link4 = root+'/1_atomic_all_p.csv'
link5 = root+'/1_logical_neg_all_p.csv'
link6 = root+'/1_semi_logical_neg_all_p.csv'

#test_data = pd.read_csv(link2)
test_data_all = pd.read_csv(link_test_all)
#annotated_data = pd.read_csv(link3)
atomic_data = pd.read_csv(link4)
anion_logical_neg_data_label_1 = pd.read_csv(link5)
anion_semi_logical_neg_data_label_1 = pd.read_csv(link6)

with open('output.txt', 'w') as file:
  print(len(test_data_all), file = file)
  print(len(atomic_data), file = file)
  print(len(anion_logical_neg_data_label_1), file = file)
  print(len(anion_semi_logical_neg_data_label_1), file = file)

link7 = root+'/0_atomic_all.csv'
link8 = root+'/0_logical_neg_all.csv'
link9 = root+'/0_semi_logical_neg_all.csv'

atomic_data_minus = pd.read_csv(link7)
anion_logical_neg_data_minus = pd.read_csv(link8)
anion_semi_logical_neg_data_minus = pd.read_csv(link9)

with open('output.txt', 'a') as file:
  print(len(atomic_data_minus), file = file)
  print(len(anion_logical_neg_data_minus), file = file)
  print(len(anion_semi_logical_neg_data_minus), file = file)

#concat all and make a new string
def concat_all_by_sep_train(example):
  output = int(example['output'])

  final_str = example['p'] + " </s> " + example['r'] + " </s> " + example['q']

  return {'label': output, 'text': final_str}

def concat_all_by_sep_train_2(example):
  output = int(example['output'])
  r = example['r']
  p = example['p']
  q = example['q']

  if r == 'oEffect':
    prompt = p + '. As a result, others then ' + q + '.'
  elif r == 'oReact':
    prompt = p + '. As a result, others feel ' + q + '.'
  elif r == 'oWant':
    prompt = p + '. As a result, others want ' + q + '.'
  elif r == 'xAttr':
    prompt = p + '. PersonX is seen as ' + q + '.'
  elif r == 'xEffect':
    prompt = p + '. As a result, PersonX then ' + q + '.'
  elif r == 'xIntent':
    prompt = p + '. Because PersonX wanted ' + q + '.'
  elif r == 'xNeed':
    prompt = p + '. Before, PersonX needed ' + q + '.'
  elif r == 'xReact':
    prompt = p + '. As a result, PersonX feels ' + q + '.'
  elif r == 'xWant':
    prompt = p + '. As a result, PersonX wants ' + q + '.'
  else:
    prompt = ''

  return {'label': output, 'text': prompt}

#checkpoint = "roberta-base"
#checkpoint = "roberta-large"
checkpoint = "facebook/bart-large"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.half()

def tokenize_function(examples):
  return tokenizer(examples["text"], padding="max_length", truncation=True)
  #return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
  metric = load_metric("accuracy")

  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def custom_metrics_all(eval_pred):
    logits, labels = eval_pred

    print('labels: ', labels)
    print('shape of labels: ', labels.shape)
    print('logits: ', logits)
    print('shape of logits: ', logits.shape)

    predictions = np.argmax(logits, axis=-1)

    print('predictions: ', predictions)

    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def custom_metrics_all_2(eval_pred):
  metric1 = load_metric("precision")
  metric2 = load_metric("recall")
  metric3 = load_metric("f1")
  metric4 = load_metric("accuracy")

  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  # micro or macro or weighted
  precision = metric1.compute(predictions=predictions, references=labels, average="weighted")["precision"]
  recall = metric2.compute(predictions=predictions, references=labels, average="weighted")["recall"]
  f1 = metric3.compute(predictions=predictions, references=labels, average="weighted")["f1"]
  accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]

  return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

# ---**Dummy classifier**---
r1 = ['a' for _ in range(5000)]
r2 = [random.randint(0, 1) for _ in range(5000)]
X1 = np.array(r1)
y1 = np.array(r2)

X2 = test_data_all['p'].values.flatten()
y2 = test_data_all['output'].values.flatten()

# Create a DummyClassifier with a "most frequent" strategy
dummy_clf1 = DummyClassifier(strategy="most_frequent")
# Fit the DummyClassifier on the training data
dummy_clf1.fit(X1, y1)
# Make predictions on the test data
y_pred1 = dummy_clf1.predict(X2)

# Create a DummyClassifier with a "most frequent" strategy
dummy_clf2 = DummyClassifier(strategy="uniform")
dummy_clf2.fit(X1, y1)
y_pred2 = dummy_clf2.predict(X2)

# Generate classification report
report1 = classification_report(y2, y_pred1)
with open('output.txt', 'a') as file:
  print("DummyClassifier(most frequent):", file = file)
  print(report1, file = file)

report2 = classification_report(y2, y_pred2)
with open('output.txt', 'a') as file:
  print("DummyClassifier(random):", file = file)
  print(report2, file = file)

#lr = 2e-5
def getTrainingArguments(size, lr_2):
  epochs = 0
  step = 0
  if size < 5100:
    epochs = 8
    step = 50
  elif size < 10001:
    epochs = 2.5
    step = 50
  else:
    epochs = 2
    step = 100

  t_args = TrainingArguments(
    output_dir='./results',          # output directory
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric

    num_train_epochs=epochs,              # total number of training epochs
    warmup_steps=step,                # number of warmup steps for learning rate scheduler
    logging_steps=step,               # log & save weights each logging_steps
    save_steps=step,

    learning_rate=lr_2,
    seed=42,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
  )
  return t_args

# .... 
size_list = [5000, 25000, 50000]
for train_size in size_list:
  # positive samples
  atomic_data = pd.read_csv(link4)
  anion_logical_neg_data_label_1 = pd.read_csv(link5)
  anion_semi_logical_neg_data_label_1 = pd.read_csv(link6)

  # negative samples
  atomic_data_minus = pd.read_csv(link7)
  anion_logical_neg_data_minus = pd.read_csv(link8)
  anion_semi_logical_neg_data_minus = pd.read_csv(link9)

  atomic_data = atomic_data.sample(frac=1, random_state=42)  # Shuffle + Set a random_state for reproducibility
  atomic_data = atomic_data.head(train_size) 
  anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.sample(frac=1, random_state=42) 
  anion_logical_neg_data_label_1 = anion_logical_neg_data_label_1.head(train_size) 
  anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.sample(frac=1, random_state=42) 
  anion_semi_logical_neg_data_label_1 = anion_semi_logical_neg_data_label_1.head(train_size) 

  atomic_data_minus = atomic_data_minus.sample(frac=1, random_state=42)
  atomic_data_minus = atomic_data_minus.head(train_size)
  anion_logical_neg_data_minus = anion_logical_neg_data_minus.sample(frac=1, random_state=42)
  anion_logical_neg_data_minus = anion_logical_neg_data_minus.head(train_size)
  anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.sample(frac=1, random_state=42)
  anion_semi_logical_neg_data_minus = anion_semi_logical_neg_data_minus.head(train_size)

  for i in range(13):
    if i == 0:
      continue
      # process ATOMIC(+)
      train_data = atomic_data
      
      with open('output.txt', 'a') as file:
        print("Dataset: ATOMIC(+), Size: ", train_size, file = file)
    elif i == 1:
      continue
      # process ANION_Logical_Neg(+)
      train_data = anion_logical_neg_data_label_1
      
      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Semi_Logical_Neg(+), Size: ", train_size, file = file)
    elif i == 2:
      continue
      # process ANION_Semi_Logical_Neg(+)
      train_data = anion_semi_logical_neg_data_label_1

      with open('output.txt', 'a') as file:
        print("Dataset: anion_semi_logical_neg(+), Size: ", train_size, file = file)
    elif i == 3:
      continue
      # process ANION_Logical_Neg(+) + ANION_Semi_Logical_Neg(+)
      train_data = pd.concat([anion_logical_neg_data_label_1, anion_semi_logical_neg_data_label_1], axis=0) #axis = 0 means row wise concatanation

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(+) + ANION_Semi_Logical_Neg(+), Size: ", train_size, file = file)
    elif i == 4:
      continue
      # process ATOMIC(-)
      train_data = atomic_data_minus

      with open('output.txt', 'a') as file:
        print("Dataset: ATOMIC(-), Size: ", train_size, file = file)
    elif i == 5:
      continue
      # logical neg (-)
      train_data = anion_logical_neg_data_minus

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 6:
      continue
      # semi logical neg (-)
      train_data = anion_semi_logical_neg_data_minus

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 7:
      continue
      # process ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(-)
      train_data = pd.concat([anion_logical_neg_data_minus, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 8:
      # process ATOMIC(+) + ATOMIC(-)
      train_data = pd.concat([atomic_data, atomic_data_minus], axis=0) #axis = 0 means row wise concatanation

      with open('output.txt', 'a') as file:
        print("Dataset: ATOMIC(+) + ATOMIC(-), Size: ", train_size, file = file)
    elif i == 9:
      continue
      # process ANION_Logical_Neg(+) + ANION_Logical_Neg(-)
      train_data = pd.concat([anion_logical_neg_data_label_1, anion_logical_neg_data_minus], axis=0)

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(+) + ANION_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 10:
      continue
      # process ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
      train_data = pd.concat([anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0) #axis = 0 means row wise concatanation

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 11:
      continue
      # process ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
      train_data = pd.concat([anion_logical_neg_data_label_1, anion_logical_neg_data_minus,
                            anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0)

      with open('output.txt', 'a') as file:
        print("Dataset: ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    elif i == 12:
      continue
      # process ATOMIC (+) + ATOMIC (-) + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-)
      train_data = pd.concat([atomic_data, atomic_data_minus, anion_logical_neg_data_label_1, anion_logical_neg_data_minus,
                            anion_semi_logical_neg_data_label_1, anion_semi_logical_neg_data_minus], axis=0)

      with open('output.txt', 'a') as file:
        print("Dataset: ATOMIC (+) + ATOMIC (-) + ANION_Logical_Neg(+) + ANION_Logical_Neg(-) + ANION_Semi_Logical_Neg(+) + ANION_Semi_Logical_Neg(-), Size: ", train_size, file = file)
    
    print('Train data size: ', len(train_data))

    td = Dataset.from_pandas(train_data)
    if '__index_level_0__' in td.column_names:
      td = td.remove_columns(['__index_level_0__'])
    
    # Filter out rows where 'q' column has value 'nan' | 'None'
    filtered_dataset = td.filter(lambda example: example['q'] != 'nan')
    filtered_dataset = filtered_dataset.filter(lambda example: example['q'] is not None)

    train_dataset = filtered_dataset.map(concat_all_by_sep_train)
    train_dataset = train_dataset.remove_columns(['p', 'q', 'r', 'output']).shuffle(seed=42)
    dts = train_dataset.train_test_split(test_size=0.10)

    test_dataset_all = Dataset.from_pandas(test_data_all)
    test_dataset_all = test_dataset_all.map(concat_all_by_sep_train)

    if '__index_level_0__' in test_dataset_all.column_names:
      test_dataset_all = test_dataset_all.remove_columns(['__index_level_0__'])
    test_dataset_all = test_dataset_all.remove_columns(['p', 'q', 'r', 'output'])

    dataset = DatasetDict()
    dataset['train'] = dts["train"]
    dataset['validation'] = dts["test"]
    dataset['test'] =  test_dataset_all

    print(dataset)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_train_dataset = small_train_dataset.select(range(1000))
    small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
    small_eval_dataset = small_eval_dataset.select(range(1000))
    
    print('small train size: ', len(small_train_dataset))
    print(type(small_train_dataset))
    print(type(small_eval_dataset))

    num_rows = small_train_dataset.num_rows
    num_columns = len(small_train_dataset.features)
    print("Shape of dataset['train']: Rows =", num_rows, ", Columns =", num_columns)

    num_rows = small_eval_dataset.num_rows
    num_columns = len(small_eval_dataset.features)
    print("Shape of dataset['eval']: Rows =", num_rows, ", Columns =", num_columns)

    lr = 2e-5
    #lr_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    lr_list = [1e-6, 5e-5]
    for each_lr in lr_list:
      tr_args = getTrainingArguments(len(small_train_dataset), each_lr)

      early_stop = EarlyStoppingCallback(3, 0.01)

      trainer = Trainer(
        model=model,
        args=tr_args,
        #train_dataset=tokenized_datasets["train"],
        train_dataset=small_train_dataset,
        #eval_dataset=tokenized_datasets["validation"],
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stop])

      trainer.train()
      trainer.evaluate()

      t = tokenized_datasets["test"].remove_columns("text")

      results = trainer.predict(t)

      preds = []
      for x in results[0]:
        y = np.argmax(x)
        preds.append(y)
      set(preds)
      actual = results[1].tolist()

      with open('output.txt', 'a') as file:
        print('lr: ', each_lr, file = file)
        print(classification_report(actual, preds))
        print(classification_report(actual, preds), file = file)

os.system("git add .")
os.system("git commit -m message")
os.system("git push")
