from transformers import pipeline
import pandas as pd

text = """I stayed at Gotia Towers and parking was a nightmare"""

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

outputs = classifier(text)
print(pd.DataFrame(outputs))
