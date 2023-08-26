import torch
import torch.nn.functional as F
import multiword_utils as utils

block_size = 3

C = torch.load('models/multiword_1_embeddings.pt')
W1 = torch.load('models/multiword_1_weights_1.pt')
b1 = torch.load('models/multiword_1_bias_1.pt')

W2 = torch.load('models/multiword_1_weights_2.pt')
b2 = torch.load('models/multiword_1_bias_2.pt')

g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all '...'
    while True:
      emb = C[torch.tensor([context])] # (1, block_size, d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)

      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    print(''.join(utils.itos[i] for i in out))
