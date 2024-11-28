#!/usr/bin/env python
# coding: utf-8

# In[1]:


from experiment_modules import *
from numpy.testing import assert_array_equal
import numpy as onp

import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from rich.progress import track


# In[2]:


generator = Generator(200)



def calculate_entropy_from_model(state, generator=generator):
    """
    Compute the entropy of a model on some data.
    """
    ntk_fn = get_ntk_function(state.apply_fn, None)

    ntk_matrix = ntk_fn(
                    generator.train_ds["inputs"],
                    generator.train_ds["inputs"],
                    {"params": state.params}
                )

    return compute_entropy(ntk_matrix)    

widths = onp.unique(onp.logspace(0, 3, 10, dtype=int))


for depth in [1, 2, 3, 4, 5]:
    entropies = {}
    for width in widths:
        for i in range(50):
            entropies[width] = {}
            model = build_network(width, depth)()
            params, losses = train_model(1000, 20, 1e-3, model, generator)

            entropies[width]["params"] = params
            entropies[width]["losses"] = losses
            onp.save(f"/data/stovey/entropy/entropies_{depth}_{i}.npy", entropies) 
