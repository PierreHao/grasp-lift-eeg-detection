import numpy as np
import pandas as pd
import sys
from glob import glob
from mne import find_events, Epochs, create_info, concatenate_raws, pick_types, compute_raw_covariance
from mne.channels import read_montage
from mne.io import RawArray

WINDOW = 500
NFILTERS = 3

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

def get_epochs_and_cov(raw_data, window=500):
    picks = range(len(getChannelNames()))
    events = list()
    events_id = dict()

    for j, eid in enumerate(getEventNames()):
        tmp = find_events(raw_data, stim_channel=eid, verbose=False)
        tmp[:, -1] = j + 1
        events.append(tmp)
        events_id[eid] = j + 1

    events = np.concatenate(events, axis=0)
    order_ev = np.argsort(events[:, 0])
    events = events[order_ev]

    epochs = Epochs(raw_data, events, events_id, 
            tmin=-(window / 500.0) + 1 / 500.0 + 0.150, 
            tmax=0.150, proj=False, picks=picks, baseline=None, 
            preload=True, add_eeg_ref=False, verbose=False) 

    cov_signal = compute_raw_covariance(draw_data, verbose=False)


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
        #print fnames

        fnames_train = fnames[:-2]
        fnames_validation = fnames[-2:]

        fnames_test = glob('data/test/subj%d_series*_data.csv' % (subject))
        fnames_test.sort()

        #print fnames_validation

        raw_train = concatenate_raws([creat_mne_raw_object(fname, read_events=True) for fname in fnames_train])
        raw_val = concatenate_raws([creat_mne_raw_object(fname, read_events=True) for fname in fnames_validation])
        #raw_test = concatenate_raws([creat_mne_raw_object(fname, read_events=False) for fname in fnames_test])

        picks_train = pick_types(raw_train.info, eeg=True)
        picks_val = pick_types(raw_val.info, eeg=True)
        #picks_test = pick_types(raw_test.info, eeg=True)

        data_train = raw_train._data[picks_train].T
        labels_train = raw_train._data[32:].T

        data_val = raw_val._data[picks_val].T
        labels_val = raw_val._data[32:].T

        #data_test = raw_test._data[picks_test].T
        #labels_test = None
        train_epochs, train_cov_signal = get_epochs_and_cov(raw_train, WINDOW)
        xd = Xdawn(n_components=NFILTERS, signal_cov=train_cov_signal, correct_overlap=False)
        xd.fit(train_epochs)

        val_epochs, val_cov_signal = get_epochs_and_cov(raw_val, WINDOW)
        xd = Xdawn(n_components=NFILTERS, signal_cov=val_cov_signal, correct_overlap=False)
        xd.fit(val_epochs)

        P = []
        for eid in getEventNames():
            P.append(np.dot(xd.filters_[eid][:, 0:NFILTERS].T, xd.evokeds_[eid].data))

        print "Saving data for subject{0} in files".format(subject)
        np.save('/data/processed/subj{0}_train.npy'.format(subject), train_epochs._data)
        np.save('/data/processed/subj{0}_val.npy'.format(subject), train_epochs.events)
