import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from modules import *


w_stds = np.linspace(0, 2., 101)
b_stds = np.linspace(0, 1., 51)

x, y = np.meshgrid(w_stds, b_stds)

coordinates = np.column_stack([x.ravel(), y.ravel()])


# In[ ]:


coordinates.shape


# In[ ]:


generator = nl.data.MNISTGenerator(500)
for coordinate in coordinates:
    main(
        w_std=coordinate[0],
        b_std=coordinate[1],
        batch_size=120,
        generator=generator 
    )
