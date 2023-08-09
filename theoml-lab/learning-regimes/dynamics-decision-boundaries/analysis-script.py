#!/usr/bin/env python
# coding: utf-8

# In[1]:

from analysis_functions import *

# In[2]:


@dataclass
class Measurement:
    width: int
    depth: int
    loss: np.ndarray
    trace: np.ndarray
    entropy: np.ndarray
    class_entropy: dict


# In[3]:


experiment_data = onp.load("linear-boundary.npy", allow_pickle=True)


# In[4]:


generator = Generator(n_samples=500)

results = []

for experiment in track(experiment_data):
    trace = []
    entropy = []
    class_entropy = []
    loss = []
    
    model = build_network(experiment.width, experiment.depth)()
    ntk_fn = get_ntk_function(model.apply, None)

    for i in range(0, len(experiment.parameters)):
        # Loss computation
        loss.append(
            ((
                model.apply(
                    {"params": experiment.parameters[i]}, generator.train_ds["inputs"]
                ) - generator.train_ds["targets"]) ** 2).mean()
        )
        
        # Compute the NTK
        ntk_matrix = ntk_fn(
                    generator.train_ds["inputs"],
                    generator.train_ds["inputs"],
                    {"params": experiment.parameters[i]}
                )
        
        # Trace computation
        trace.append(
            compute_trace(ntk_matrix)
        )
        
        # Entropy computation
        entropy.append(
            compute_entropy(ntk_matrix)
        )
        
        # Class entropy computation
        class_entropy.append(
            compute_class_entropy(generator.train_ds, ntk_matrix)
        )
    results.append(
        Measurement(
            width=experiment.width,
            depth=experiment.depth,
            loss=np.array(loss),
            trace=np.array(trace),
            entropy=np.array(entropy),
            class_entropy=class_entropy
        )
    )
    
np.save("experiment-analysis.npy", results)

