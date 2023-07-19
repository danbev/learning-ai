from datasets import list_datasets
from datasets import load_dataset
#from huggingface_hub import list_datasets

ds_sets = list_datasets()
print(f'Number of datasets: {len(ds_sets)}')
print(f"The first 10 are: {ds_sets[:10]}")

# Load data set emotion
emotions = load_dataset("emotion")
emotions = load_dataset("dair-ai/emotion")

# Print emotion
print(emotions)

train_ds = emotions['train']
print(f'Columns: {train_ds.column_names}')
print(train_ds)
print(train_ds[0])
print(train_ds[1])
print(train_ds[1]["text"])
print(train_ds[1]["label"])
