from transformers import pipeline
import pandas as pd

text = """I stayed at Gotia Towers and parking was a nightmare. I would like a refund"""

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

outputs = classifier(text)
print(pd.DataFrame(outputs))


ner_tagger = pipeline("ner", aggregation_strategy="simple", model="dbmdz/bert-large-cased-finetuned-conll03-english")
outputs = ner_tagger(text)
print(pd.DataFrame(outputs))


reader = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What does the customer want?"
outputs = reader(question=question, context=text, model="distilbert-base-uncased-distilled-squad")
print(pd.DataFrame([outputs]))


"""
summarizer = pipeline("summarization", model="distilbart-cnn-12-6")
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
"""
