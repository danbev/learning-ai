import numpy as np
import matplotlib.pyplot as plt

def pos_encoding(seq_len, d):
    enc = np.zeros(shape=(seq_len, d))
    for k in range(seq_len):
        for i in np.arange(stop=int(d/2)):
            denominator = np.power(10000, 2*i/d)
            enc[k, 2*i] = np.sin(k/denominator)
            enc[k, 2*i+1] = np.cos(k/denominator)
    return enc

enc = pos_encoding(seq_len=4, d=4)
print(enc)
