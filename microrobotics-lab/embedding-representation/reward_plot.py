import numpy as np
import matplotlib.pyplot as plt

data = np.load('exploration.npy', allow_pickle=True)

plt.plot(data)
plt.show()