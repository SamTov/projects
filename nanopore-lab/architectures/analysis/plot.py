import numpy as np
import matplotlib.pyplot as plt

data = np.load("confusion_matrix.npy", allow_pickle=True)

plt.imshow(data / data.sum())
plt.savefig("me.png")
