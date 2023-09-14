#!/usr/bin/env python
# coding: utf-8

# # Event labelling

# ## Histogram, peak finding and fitting

# The first part deals with the necessary methods for creating histograms, finding the peaks in those histograms and fitting the histograms to Voigt peaks.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


import scipy.signal as signal
from lmfit.models import VoigtModel


# In[ ]:


def compute_hist(data: list, n_bins: int):
    """Compute the histogram of a given list."""
    # Sanity check
    assert np.isfinite(data).all()
    assert len(data) != 0

    hist, bins = np.histogram(data, n_bins)
    bins = (bins[:-1] + bins[1:]) / 2
    return bins, hist


# In[ ]:


def evaluate_histogram(hist: np.ndarray, bins: np.ndarray, num_peaks: int):
    """Fit Voigt peaks to the histograms.

    Does the following:
     - Uses scipy.signal.find_peaks to calculate initial peaks, variies the algorithm parameters in order to
     find the amount of peaks given in num_models. This is prone to errors, so one needs to manually verify
     the initial guesses for the peaks!
     - Uses lmfit to fit the number of given peaks as Voigt peaks. Also error prone, so check that the initial fit
     looks good!

    Parameters
    ----------
    hist : np.ndarray
    bins : np.ndarray
    num_peaks : int
        Number of peaks to expect in the histogram.
    """
    model = VoigtModel(prefix="Model1_")
    if num_peaks > 1:
        for mod_id in range(2, num_peaks + 1):
            model += VoigtModel(prefix=f"Model{mod_id}_")

    # Produce initial guesses for the peaks
    incr_height = 100
    incr_prominence = 60
    peaks, _ = signal.find_peaks(
        hist, height=incr_height, prominence=incr_prominence, wlen=50, distance=5
    )
    peaks = sorted(peaks, reverse=True)

    #     plt.plot(bins, hist)
    #     plt.plot(bins[peaks], hist[peaks], '.')
    #     plt.show()

    # If less than required number of peaks is found, relax peak conditions and try again
    # play around with these parameters if wrong peaks are found
    wlen = 50
    distance = 5
    while len(peaks) < num_peaks:
        incr_height -= 10
        incr_prominence -= 10
        peaks, _ = signal.find_peaks(
            hist,
            height=incr_height,
            prominence=incr_prominence,
            wlen=wlen,
            distance=distance,
        )
        peaks = sorted(peaks, reverse=True)

    # Sanity check
    plt.plot(bins, hist)
    plt.plot(bins[peaks], hist[peaks], ".")
    plt.show()

    # Initialize models, restrict fit parameters to be within initial regime
    for mod_id in range(1, num_peaks + 1):
        model.set_param_hint(
            f"Model{mod_id}_center",
            value=bins[peaks[mod_id - 1]],
            min=bins[peaks[mod_id - 1]] - 0.1,
            max=bins[peaks[mod_id - 1]] + 0.1,
        )
        model.set_param_hint(f"Model{mod_id}_sigma", value=0.005, min=0)
        model.set_param_hint(
            f"Model{mod_id}_amplitude", value=hist[peaks[mod_id - 1]] / 50, min=0
        )

    params = model.make_params()
    result = model.fit(hist, x=bins, params=params)
    return result


# ## Labelling the data

# In[ ]:


import pickle

import pandas as pd


# ## Loading the experimental data
# The experimental data contains a tuple of two lists, the first containing the normalized events and the second containing the ratio of mean current inside the event and the surrounding baseline current.
# 
# The latter list is then used for producing the histograms.

# In[ ]:


experiment_data_dir = "/data/jhossbach/Experiments/"

# For reading the files
indexs = list()
for i in range(1, 7):
    for j in range(4, 11):
        indexs.append(f"L{i}AS{j}")

# Sort by ladder
list_dict = {}
for i in range(1, 7):
    list_dict[f"Ladder_{i}"] = ([], [])

for index in indexs:
    with open(experiment_data_dir + index + ".pk", "rb") as f:
        data = pickle.load(f)
        list_dict[f"Ladder_{index[1]}"][0].extend(data[0])
        list_dict[f"Ladder_{index[1]}"][1].extend(data[1])

df = pd.DataFrame(list_dict, index=["outer_events", "ib_i0"])


# This df now contains two rows for the  outer events (a list of lists) and the ratio ib/i0 (a list of floats).
# 
# Each column represents the combined data of all measurements of one ladder.

# In[ ]:


print(np.isfinite(df["Ladder_2"].ib_i0).all())
for event in df["Ladder_2"].outer_events:
    assert np.isfinite(event).all()


# In[ ]:


for i in np.random.randint(0, 100, (10,)):
    plt.plot(df["Ladder_5"].outer_events[i])
    plt.hlines(
        df["Ladder_5"].ib_i0[0], 0, len(df["Ladder_5"].outer_events[i]), colors=["C2"]
    )
    plt.show()


# ## Labelling based on the fits
# 
# Labelling the events is now done by computing the fits and assigning the labels for all events that fall into the the interval `[maximum-.5*FWHM, maximum+.5*FWHM]`

# In[ ]:


events_dict = {}
res_dict = {}
for ladder, series in df.items():
    print(f"Looking at {ladder}")
    # Compute the histogram and produce the fits
    assert np.isfinite(series.ib_i0).all()
    bins, hist = compute_hist(series.ib_i0, 260)
    res = evaluate_histogram(hist, bins, num_peaks=7)
    res_dict[ladder] = res

    # Assign labels for all events inside interval around peaks
    upper_list = []
    lower_list = []
    for mod_id in range(1, 8):
        # Short name of peptide (eg. L1AS4)
        peptide_name = f"{ladder.replace('adder_', '')}AS{mod_id+3}"

        # Retrieve parameters
        center = res.best_values[f"Model{mod_id}_center"]
        fwhm = res.params[f"Model{mod_id}_fwhm"].value
        upper = center + 0.5 * fwhm
        lower = center - 0.5 * fwhm
        upper_list.append(upper)
        lower_list.append(lower)

        # Sanity checking
        plt.vlines(lower, ymin=0, ymax=max(hist), linestyles="dashed", colors=["C2"])
        plt.vlines(upper, ymin=0, ymax=max(hist), linestyles="dashed", colors=["C2"])
        indxs = np.argwhere(
            np.logical_and(
                series.ib_i0 > center - 0.5 * fwhm, series.ib_i0 < center + 0.5 * fwhm
            )
        )
        events = np.array(series.outer_events, dtype=object)[indxs].squeeze()
        events_dict[peptide_name] = events

        _len = len(events_dict[peptide_name])
        print(f"Number of labeled events for {peptide_name}: {_len}")

    # Sanity check two see if regions overlap (https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap/325964#325964)
    for i in range(len(upper_list), -1):
        assert lower_list[i] < upper_list[i + 1] and lower_list[i + 1] > upper_list[i]

    plt.plot(bins, hist)
    plt.plot(bins, res.best_fit)
    plt.plot(bins, res.init_fit, "--")
    plt.show()
df.loc["res"] = res_dict


# ## Saving the data
# 
# Each pickle file in this directory will contain all events that were labelled to a specific sequence, so e.g. `L1AS4.pk` contains all events of the sequence `L1AS4`

# In[ ]:


import pathlib
import pickle

path = pathlib.Path("/data/jhossbach/Peptide_events")

# Uncomment to overwrite
# for name, events in events_dict.items():
#     with open(path / f"{name}.pk", "wb") as f:
#         pickle.dump(events, f)


# In[ ]:




