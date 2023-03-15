from transformers import pipeline, DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
pipe = pipeline("text-classification", "./snips_clf/results", tokenizer=tokenizer)
print(pipe('Coco'))