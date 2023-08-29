#!/usr/bin/env python
# coding: utf-8

# # Methods for doing the event detection and refining the events

# In[ ]:


import numpy as np
from event_analysis.analysis.event_detection import AmplitudeThreshold, EventDetection


# In[ ]:


def find_raw_events(
    time_series: np.ndarray, start_threshold: int, mean_threshold: int
) -> list[slice]:
    """Return the events as found by the Amplitude Threshold method.

    Parameters
    ----------
    time_series : np.ndarray
    start_threshold, mean_threshold : int
        Start and mean thresholds for the AmplitudeThreshold method

    Returns
    -------
    A list of slices as found by the Amplitude Threshold for the given time series.
    """
    amplitude_threshold = AmplitudeThreshold(
        start_threshold=start_threshold, mean_threshold=mean_threshold
    )
    init_event_slices = amplitude_threshold(time_series)
    return init_event_slices


# In[ ]:


def find_events(
    time_series: np.ndarray,
    start_threshold: int = 4,
    mean_threshold: int = 5,
    min_event_length: int = 80,
) -> list:
    """Refine the events.

    Does the following:
        - the baseline_current_means method computes the mean base current around the event
        - the include_event_rise_and_descend method fine tunes the edges of the event
        - the exclude_event_rise_and_descend method is used to compute the mean current of the event,
           which is later used to calculate I/I_0.
    Also asserts that none of the events contain non-finite values and are greater than zero.

    Parameters
    ----------
    time_series: np.ndarray
    start_threshold, mean_threshold : int
        Start and mean thresholds for the AmplitudeThreshold method
    min_event_length : int
        Minimum length of an event to be counted as an actual event.

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
    """
    # Detect events
    init_event_slices = find_raw_events(
        time_series=time_series,
        start_threshold=start_threshold,
        mean_threshold=mean_threshold,
    )
    event_slices, mean_currents = EventDetection.baseline_current_means(
        time_series, init_event_slices, min_length=min_event_length, window_size=5000
    )
    inner_slices = EventDetection.exclude_event_rise_and_descend(
        time_series=time_series,
        event_slices=event_slices,
    )
    outer_slices = EventDetection.include_event_rise_and_descend(
        time_series=time_series,
        event_slices=event_slices,
        region_extension=40,
    )

    # Sanity check
    assert (
        len(event_slices)
        == len(mean_currents)
        == len(inner_slices)
        == len(outer_slices)
    )

    # Combine slices
    combined_events = []
    for i in range(len(event_slices)):
        combined_events.append(
            [event_slices[i], inner_slices[i], outer_slices[i], mean_currents[i]]
        )

    # Remove edge cases
    def detect_edge_case(event_tuple):
        is_none = any([event is None for event in event_tuple])
        if not is_none:
            is_zero = any([event.stop - event.start == 0 for event in event_tuple[:-1]])
        return is_none or is_zero

    for event_tuple in combined_events:
        if detect_edge_case(event_tuple):
            combined_events.remove(event_tuple)

    # Manually add space around outer_slices until include_rise_and_descend works better
    for event_tuple in combined_events:
        event_length = event_tuple[2].stop - event_tuple[2].start
        additional_padding = int(0.35 * event_length)
        _slice = np.s_[
            event_tuple[2].start
            - additional_padding : event_tuple[2].stop
            + additional_padding
        ]
        event_tuple[2] = _slice

    # Get events
    inner_events = [time_series[event_tuple[1]] for event_tuple in combined_events]
    outer_events = [time_series[event_tuple[2]] for event_tuple in combined_events]

    # Normalization using the mean baseline current
    normed_events = [
        event / combined_events[i][3] for i, event in enumerate(outer_events)
    ]
    ib_i0 = [
        np.mean(event) / combined_events[i][3] for i, event in enumerate(inner_events)
    ]

    # Yet another sanity check
    faulty_event_indxs = []
    for event_list in (normed_events, outer_events, inner_events):
        for index in range(len(event_list)):
            try:
                assert event_list[index] is not None
                assert np.isfinite(event_list[index]).all()
                assert not np.isnan(event_list[index]).any()
                assert len(event_list[index]) != 0
            except AssertionError:
                print(f"Removing event {index} that did not survive sanity check")
                faulty_event_indxs.append(index)

    # Remove recursively
    for indx in reversed(sorted(faulty_event_indxs)):
        for event_list in (normed_events, outer_events, inner_events):
            del event_list[indx]
        del ib_i0[indx]

    # Final Sanity check after removing
    assert all(
        [
            len(ib_i0) == len(event_list)
            for event_list in (normed_events, outer_events, inner_events)
        ]
    )

    return normed_events, ib_i0


# # Iterate through all data and produce events, safe in per-experiment datasets

# In[ ]:


import copy
import gc
import pickle

import matplotlib.pyplot as plt
from event_analysis.datasets import LadderDataset


# In[ ]:


sql_url = "mysql+pymysql://ladder:EHBiaLTVm872gx7R@129.69.120.127/ladderds"


# In[ ]:


ladders = LadderDataset.all(
    name="All data", data_path="/data/jhossbach/LadderData/Ladder.hdf5", sql_url=sql_url
)


# In[ ]:


ladder_dict = {name: ex.time_series for name, ex in ladders.experiments.items()}


# In[ ]:


test_events = find_events(ladder_dict["L2AS9"]())


# In[ ]:


for i in np.random.choice(len(test_events[0]), size=10):
    plt.plot(test_events[0][i])
    plt.show()


# In[ ]:


list_ = list()
for name, time_series in ladder_dict.items():
    print(f"Looking at {name}")
    data = copy.deepcopy(time_series())
    assert isinstance(data, np.ndarray)
    normed_events, ib_i0 = find_events(data)

    plt.hist(ib_i0, bins=200)
    plt.show()

    with open(f"/data/jhossbach/Experiments/{name}.pk", "wb") as f:
        pickle.dump((normed_events, ib_i0), f)
    del time_series
    del data
    gc.collect()


# In[ ]:


for name, time_series in ladder_dict.items():
    with open(f"/data/jhossbach/Experiments/{name}.pk", "rb") as f:
        try:
            normed_events, ib_i0 = pickle.load(f)
            for event in normed_events:
                assert np.isfinite(event).all()
            assert np.isfinite(ib_i0).all()
        except AssertionError:
            print(f"Error in file {name}.pk")


# In[ ]:




