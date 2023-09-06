import numpy as np

häagen_dazs = np.array([0.81, 0.77, 0.95])
mövenpick = np.array([0.72, 0.70, 0.93])
dumbell = np.array([0.91, 0.11, 0.25])

def cosine_similarity(w1, w2):
  return np.dot(w1,w2) / (np.dot(w1,w1) * np.dot(w2,w2))**0.5

print(f'{cosine_similarity(häagen_dazs, mövenpick)=}')
print(f'{cosine_similarity(häagen_dazs, dumbell)=}')
print(f'{cosine_similarity(mövenpick, dumbell)=}')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, häagen_dazs[0], häagen_dazs[1], häagen_dazs[2], color='b', label='häagen_dazs')
ax.quiver(0, 0, 0, mövenpick[0], mövenpick[1], mövenpick[2], color='r', label='mövenpick')
ax.quiver(0, 0, 0, dumbell[0], dumbell[1], dumbell[2], color='g', label='dumbell')

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()
plt.show()

