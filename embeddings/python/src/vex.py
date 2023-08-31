import numpy as np
from transformers import BertTokenizer, TFBertModel


def cos_sim(x, y):
    numerator = np.dot(x, y)

	#Taking root of squared sum of x and y
    x_normalised = np.sqrt(np.sum(x**2))
    y_normalised = np.sqrt(np.sum(y**2))

    denominator = x_normalised * y_normalised
    cosine_similarity = numerator / denominator
    return cosine_similarity

model_name="prajjwal1/bert-small"
tokenizer=BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name,from_pt=True)
def encode_sentences(sentences):
  encoded = tokenizer.batch_encode_plus(
                      [sentences],
                      max_length=512,
                      pad_to_max_length=False,
                      return_tensors="tf",
                  )

  input_ids = np.array(encoded["input_ids"], dtype="int32")
  output = model(
      input_ids
  )
  sequence_output, pooled_output = output[:2]
  return pooled_output[0]

vex = """
"title": "Red Hat Security Advisory: Red Hat AMQ Broker 7.8.7 release and security update",
"id": "RHSA-2022:6292",
"cve": "CVE-2022-35278",
"cwe": {"id": "CWE-74"}
"""
print(f'{vex=}')

prod_name = "Red Hat JBoss AMQ Broker"
print(f'{prod_name=}')
embed1=encode_sentences(vex)
embed2=encode_sentences(prod_name)

cosign_similarity = cos_sim(embed1, embed2)
print("Cosine similarity Score {}".format(cosign_similarity))
