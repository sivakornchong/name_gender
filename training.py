import tensorflow
from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification, DistilBertTokenizerFast, \
     DataCollatorWithPadding, pipeline
from datasets import load_metric, Dataset
import numpy as np

female_file = open('Data/female.txt', 'rb')
female_rows = female_file.readlines()
# print(female_rows[:20])

male_file = open('Data/male.txt', 'rb')
male_rows = male_file.readlines()

# This code segment parses the snips dataset into a more manageable format

utterances = []
# tokenized_utterances = []
labels_for_tokens = []
sequence_labels = []

###Female = 1, Male = 0

for snip_row in female_rows:
     name = snip_row.decode().split('\n')[0].lower()
     utterances.append(name)
     name_label = 1
     sequence_labels.append(name_label)

for snip_row in male_rows:
     name = snip_row.decode().split('\n')[0].lower()
     utterances.append(name)
     name_label = 0
     sequence_labels.append(name_label)

# print(utterances[0])
# print(sequence_labels[0])
#
# print(len(utterances))
# print(len(sequence_labels))

gender_dataset = Dataset.from_dict(  # hold data for both sequence and token classification
    dict(
        utterance=utterances,
        label=sequence_labels,
    )
)

##This code segment tokenize the dataset for training
gender_dataset = gender_dataset.train_test_split(test_size=0.2)

print(len(gender_dataset['train']))
print(gender_dataset['train'][0])

# simple function to batch tokenize utterances with truncation
def preprocess_function(examples):
    return tokenizer(examples["utterance"], truncation=True)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
seq_clf_tokenized_snips = gender_dataset.map(preprocess_function, batched=True)
print(seq_clf_tokenized_snips['train'][50])

# DataCollatorWithPadding creates batch of data. It also dynamically pads text to the
#  length of the longest element in the batch, making them all the same length.
#  It's possible to pad your text in the tokenizer function with padding=True, dynamic padding is more efficient.

##This code segment tokenize the model for training

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
sequence_clf_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2,
)
# set an index -> label dictionary
sequence_clf_model.config.id2label = {0: 'Male', 1: 'Female'}

#Creating accuracy as a new matrix on top of losses to be monitored
metric = load_metric("accuracy")
def compute_metrics(eval_pred):  # custom method to take in logits and calculate accuracy of the eval set
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

epochs = 3
training_args = TrainingArguments(
    output_dir="./snips_clf/results",
    num_train_epochs=epochs,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    load_best_model_at_end=True,

    # some deep learning parameters that the Trainer is able to take in
    warmup_steps=len(seq_clf_tokenized_snips['train']) // 5,  # number of warmup steps for learning rate scheduler,
    weight_decay=0.05,

    logging_steps=1,
    log_level='info',
    evaluation_strategy='epoch',
    save_strategy='epoch'
)

# Define the trainer:

trainer = Trainer(
    model=sequence_clf_model,
    args=training_args,
    train_dataset=seq_clf_tokenized_snips['train'],
    eval_dataset=seq_clf_tokenized_snips['test'],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

print('initial model trainer performance is:', trainer.evaluate())

###FINALLY TRAIN THE MODEL
trainer.train()
trainer.save_model()