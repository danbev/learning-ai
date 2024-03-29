from datasets import list_datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
#from huggingface_hub import list_datasets

#from huggingface_hub import HfApi
#api = HfApi()

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
print(f'Features(datatypes: {train_ds.features}')
print(train_ds)
print(train_ds[0])
print(train_ds[1])
print(train_ds[1]["text"])
print(train_ds[1]["label"])


print(train_ds[:5])

print(train_ds["label"][:5])
print(type(train_ds["label"][:5]))

emotions.set_format(type="pandas")
df = emotions["train"][:]
print(df.head())

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

# The label_name will become a new column in the output dataframe
df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False,
          showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()


# Reset the format from pandas to the default
emotions.reset_format()

# Tokens section
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)
