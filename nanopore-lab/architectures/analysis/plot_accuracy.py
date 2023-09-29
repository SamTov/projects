import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("validation_accuracy")
print(max(data[:, 1]))
plt.plot(data[:, 1])
plt.show()
