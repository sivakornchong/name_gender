from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification, DistilBertTokenizerFast, \
     DataCollatorWithPadding, pipeline
from datasets import load_metric, Dataset
import numpy as np

female_file = open('Data/female.txt', 'rb')
snips_rows = female_file.readlines()
print(snips_rows[:20])

# male_file = 