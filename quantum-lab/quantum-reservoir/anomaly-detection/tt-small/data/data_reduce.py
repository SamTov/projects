import pandas as pd

back = pd.read_csv("background.csv", delimiter = ";")
sig = pd.read_csv("signal.csv", delimiter = ";")


back_small = back[["MET", "b1 pT"]]
sig_small = sig[["MET", "b1 pT"]]

back_small.to_csv("background_small.csv")
sig_small.to_csv("signal_small.csv")
