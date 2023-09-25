#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#get_ipython().run_line_magic('run', 'experiment-modules.ipynb')
from modules import *

# In[ ]:


w_stds = np.linspace(0, 1., 51)
b_stds = np.linspace(0, 1., 51)

x, y = np.meshgrid(w_stds, b_stds)

coordinates = np.column_stack([x.ravel(), y.ravel()])


# In[ ]:


coordinates.shape


# In[ ]:


generator = nl.data.MNISTGenerator(100)
for coordinate in coordinates:
    main(
        w_std=coordinate[0],
        b_std=coordinate[1],
        batch_size=20,
        generator=generator 
    )


# In[ ]:




