from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

learn = load_learner('models/smulan-model.pkl')

smulan = PILImage.create('src/smulan.jpg')
smulan.to_thumb(192)

cat,_, probs = learn.predict(smulan)
print(f"Is smulan a cat?: {cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
