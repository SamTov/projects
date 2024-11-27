import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import sys

base_path = sys.argv[1]

template = f"{base_path}/rdf.xvg"

data = np.loadtxt(template, comments=["#", "@"])

kb = 4 * np.pi * cumtrapz(y=data[:, 0] ** 2 * (data[:, 1] - 1), x=data[:, 0])

np.save("kb.npy", kb)
