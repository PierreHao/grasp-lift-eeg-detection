import numpy as np
import pandas as pd
from glob import glob
from mne import concatenate_raws
from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info, concatenate_raws, pick_types
from sklearn.base import BaseEstimator, TransformerMixin
from glob import glob


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


subjects = range(1, 13)

lbls_tot = []
subjects_val_tot = []
series_val_tot = []

ids_tot = []
subjects_test_tot = []
series_test_tot = []

for subject in subjects:
    print 'Loading data for subject %d...' % subject
    # ############### READ DATA ###############################################
    fnames = glob('data/train/subj%d_series*_data.csv' % (subject))
    fnames.sort()
    fnames_val = fnames[-2:]

    fnames_test = glob('data/test/subj%d_series*_data.csv' % (subject))
    fnames_test.sort()

    raw_val = concatenate_raws([create_mne_raw_object(fname, read_events=True)
                                for fname in fnames_val])
    raw_test = concatenate_raws([create_mne_raw_object(fname, read_events=False)
                                for fname in fnames_test])

    # extract labels for series 7&8
    labels = raw_val._data[32:]
    lbls_tot.append(labels.transpose())

    # aggregate infos for validation (series 7&8)
    raw_series7 = create_mne_raw_object(fnames_val[0])
    raw_series8 = create_mne_raw_object(fnames_val[1])
    series = np.array([7] * raw_series7.n_times +
                      [8] * raw_series8.n_times)
    series_val_tot.append(series)

    subjs = np.array([subject]*labels.shape[1])
    subjects_val_tot.append(subjs)

    # aggregate infos for test (series 9&10)
    ids = np.concatenate([np.array(pd.read_csv(fname)['id'])
                         for fname in fnames_test])
    ids_tot.append(ids)
    raw_series9 = create_mne_raw_object(fnames_test[1], read_events=False)
    raw_series10 = create_mne_raw_object(fnames_test[0], read_events=False)
    series = np.array([10] * raw_series10.n_times +
                      [9] * raw_series9.n_times)
    series_test_tot.append(series)

    subjs = np.array([subject]*raw_test.n_times)
    subjects_test_tot.append(subjs)


# save validation infos
subjects_val_tot = np.concatenate(subjects_val_tot)
series_val_tot = np.concatenate(series_val_tot)
lbls_tot = np.concatenate(lbls_tot)
toSave = np.c_[lbls_tot, subjects_val_tot, series_val_tot]
np.save('infos_val.npy', toSave)

# save test infos
subjects_test_tot = np.concatenate(subjects_test_tot)
series_test_tot = np.concatenate(series_test_tot)
ids_tot = np.concatenate(ids_tot)
toSave = np.c_[ids_tot, subjects_test_tot, series_test_tot]
np.save('infos_test.npy', toSave)

