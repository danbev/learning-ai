from transformers import AutoTokenizer, AutoModel                               
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")                    
model = AutoModel.from_pretrained("bert-base-cased")                            
input_ids = tokenizer("cat", return_tensors="pt")["input_ids"]                  
embeddings = model(input_ids)[0]
print(f'{embeddings.shape=}')
print(f'{embeddings=}')
