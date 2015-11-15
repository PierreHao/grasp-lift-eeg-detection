import numpy as np
import pandas as pd
import sys
from glob import glob
from mne import find_events, Epochs
from mne import create_info, concatenate_raws, pick_types
from mne.channels import read_montage
from mne.io import RawArray


def getChannelNames():
    """Return Channels names."""
    return ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 
            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 
            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 
            'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']

def getEventNames():
    """Return Event name."""
    return ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff',
            'Replace', 'BothReleased']


def creat_mne_raw_object(fname, read_events = True):

    print "loading data from %s" %fname
    data = pd.read_csv(fname)

    ch_names = list(data.columns[1:])

    montage = read_montage('standard_1005', ch_names)
    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T
    
    if read_events:
        ev_fname = fname.replace('_data', '_events')
        print ev_fname
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T

        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        data = np.concatenate((data, events_data))
        
    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type, montage=montage)
    info['filename'] = fname
    raw = RawArray(data, info, verbose=True)

    return raw
        
if __name__ == '__main__':
    subjects = range(1,13)

    for subject in subjects:
        print "Loading data for subject %d... " % subject

        fnames = glob('data/train/subj%d_series*_data.csv' % (subject))
        fnames.sort()
        print fnames

        fnames_train = fnames[:-2]
        fnames_validation = fnames[-2:]

        fnames_test = glob('data/test/subj%d_series*_data.csv' % (subject))
        fnames_test.sort()

        print fnames_validation

        raw_train = concatenate_raws([creat_mne_raw_object(fname, read_events=True) for fname in fnames_train])
        raw_val = concatenate_raws([creat_mne_raw_object(fname, read_events=True) for fname in fnames_validation])
        raw_test = concatenate_raws([creat_mne_raw_object(fname, read_events=False) for fname in fnames_test])

        picks_train = pick_types(raw_train.info, eeg=True)
        picks_val = pick_types(raw_val.info, eeg=True)
        picks_test = pick_types(raw_test.info, eeg=True)

        data_train = raw_train._data[picks_train].T
        labels_train = raw_train._data[32:].T

        data_val = raw_val._data[picks_val].T
        labels_val = raw_val._data[32:].T

        data_test = raw_test._data[picks_test].T
        labels_test = None


