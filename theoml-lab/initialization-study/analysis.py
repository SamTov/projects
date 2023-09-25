#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import glob
from dataclasses import dataclass
import h5py as hf


# In[18]:


@dataclass
class Measurement:
    entropy: float = None
    trace: float = None
    train_loss: np.ndarray = None
    test_loss: np.ndarray = None
    train_accuracy: np.ndarray = None,
    test_accuracy: np.ndarray = None,
    w_std: float = None
    b_std: float = None


# In[24]:


file_list = glob.glob("/data/stovey/initialization/Dense/*.h5")


# In[44]:


results = {}
for item in file_list:
    parts = item.split("/")[-1].split("_")
    name = f"{parts[0]}_{parts[1]}"
    
    if name not in list(results):
        results[name] = Measurement()
        results[name].w_std = parts[0]
        results[name].b_std = parts[1]
        
    if parts[2] == "train":
        with hf.File(item, "r") as db:
            loss = db["loss"][:]
            accuracy = db["accuracy"][:]
            
        results[name].train_loss = loss
        results[name].train_accuracy = accuracy

        
    elif parts[2] == "test":
        with hf.File(item, "r") as db:
            loss = db["loss"][:]
            accuracy = db["accuracy"][:]
            
        results[name].test_loss = loss
        results[name].test_accuracy = accuracy
  
    elif parts[2] == "cv":
        with hf.File(item, "r") as db:
            entropy = db["entropy"][:]
            trace = db["trace"][:]
            
        results[name].entropy = entropy
        results[name].trace = trace
            


# In[46]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))


for item in results:
    ax[0].plot(results[item].trace, min(results[item].test_loss), '.')
    ax[1].plot(results[item].entropy, min(results[item].test_loss), '.')
    
ax[0].set_xscale("log")
ax[0].set_yscale("log")

ax[0].set_xlabel("Starting Trace")
ax[0].set_ylabel("Minimum Test Loss")

# ax[0].set_xscale("log")
ax[1].set_yscale("log")
# ax[1].set_xscale("log")

ax[1].set_xlabel("Starting Entropy")

plt.show()


# In[63]:


fig, ax = plt.subplots(figsize=(10, 5))
# ax2 = ax.twinx()

data = []
weight_stuff = []
for item in results:
    data.append([results[item].trace, results[item].entropy, min(results[item].test_loss)])
#     data.append([results[item].trace, results[item].entropy, results[item].test_loss[-1]])
#     data.append([results[item].trace, results[item].entropy, results[item].test_accuracy[-1]])
    #data.append([results[item].trace, results[item].entropy, max(results[item].test_accuracy)])
    weight_stuff.append([
        results[item].w_std,
        results[item].b_std,
        results[item].trace, 
        results[item].entropy, 
        min(results[item].train_loss)
    ])

data = np.array(data)
weight_stuff = np.array(weight_stuff)

im = ax.scatter(
    data[:, 0], 
    data[:, 1], 
    c=data[:, 2],
    cmap="tab20",
    norm=matplotlib.colors.LogNorm()
)
ax.set_xscale("log")
# ax.set_yscale("log")

# ax2.plot(data[:, 0], data[:, 2], '.')
# # ax2.set_yscale("log")
# ax2.set_ylabel("Train Loss")

cbar_ax = fig.add_axes([0.25, 0.95, 0.5, 0.01])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
# cbar.set_label(r"Test Loss", fontsize=12, loc="left")

#plt.savefig("dense-plot.pdf")
ax.set_xlabel("Trace")
ax.set_ylabel("Entropy")
plt.savefig("dense-plot.pdf")

plt.show()


# In[60]:


weights = weight_stuff[:, 0]
biases = weight_stuff[:, 1]
trace = data[:, 0]
entropy = data[:, 1]
loss = data[:, 2]


# In[61]:


X, Y = np.meshgrid(trace, entropy)
Z = np.meshgrid(loss, loss)
fig, ax = plt.subplots()

im = ax.pcolormesh(
    X.astype(float), 
    Y.astype(float), 
    Z[0].astype(float),
    cmap="tab20",
    norm=matplotlib.colors.LogNorm(),
    shading="gouraud"
)

fig.colorbar(im, ax=ax)

plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Trace")
plt.ylabel("Entropy")
plt.show()


# In[65]:


fig, ax = plt.subplots(1, 3, figsize=(15, 5))

X, Y = np.meshgrid(weights, biases)
y_choices = [entropy, trace, loss]
titles = ["Entropy", "Trace", "Train Loss"]

for i, choice in enumerate(y_choices):

    
    Z = np.meshgrid(choice, choice)
    
    if i == 0:
        norm=None
    else:
        norm=matplotlib.colors.LogNorm()

    im = ax[i].pcolormesh(
        X.astype(float), 
        Y.astype(float), 
        Z[0].astype(float),
        cmap="tab20",
        norm=norm,
        shading="gouraud"
    )

    fig.colorbar(im, ax=ax[i])
    ax[i].set_xlabel(r"$\sigma^{weight}$")
    ax[i].set_ylabel(r"$\sigma^{bias}$")
    ax[i].set_title(titles[i])

plt.show()


# In[ ]:




