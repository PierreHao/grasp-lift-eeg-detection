import numpy as np
import pandas as pd
from glob import glob
from mne import concatenate_raws
from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info, concatenate_raws, pick_types
from sklearn.base import BaseEstimator, TransformerMixin


def create_mne_raw_object(fname, read_events=True):
    """Create a mne raw instance from csv file."""
    # Read EEG file
    data = pd.read_csv(fname)
    print fname, data.shape

    # get chanel names
    ch_names = list(data.columns[1:])
    print ch_names

    # read EEG standard montage from mne
    montage = read_montage('standard_1005', ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T

    if read_events:
        # events file
        ev_fname = fname.replace('_data', '_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T

        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data, events_data))

    # create and populate MNE info structure
    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type,
                       montage=montage)
    info['filename'] = fname

    # create raw object
    raw = RawArray(data, info, verbose=False)

    return raw
