#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mdsuite as mds
from mdsuite.utils import Units

import glob


# In[2]:


units = Units(
        time=1,
        length=1,
        energy=1,
        NkTV2p=1,
        boltzmann=1,
        temperature=1,
        pressure=1,
    )


# In[3]:


project = mds.Project("LJ")


# In[4]:


sim_directories = glob.glob("lj_*")


# In[5]:


sim_directories


# In[6]:


for item in sim_directories:
    project.add_experiment(
        name=item,
        timestep=0.005,
        temperature=float(item.split("_")[1]),
        units=units,
        simulation_data=f"{item}/dumpy.lammpstraj"
    )


# In[12]:


project.run.RadialDistributionFunction(number_of_configurations=500, batches=5)


# In[13]:


project.experiments.lj_2.run_visualization()


# In[ ]:




