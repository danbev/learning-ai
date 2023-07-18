from datasets import list_datasets
from datasets import load_dataset

ds_sets = list_datasets()
print(f'Number of datasets: {len(ds_sets)}')
print(f"The first 10 are: {ds_sets[:10]}")

# Load data set emotion
emotions = load_dataset("emotion")
emotions = load_dataset("dair-ai/emotion")

# Print emotion
print(emotions)
