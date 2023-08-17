import scipy.io as scipy
import numpy as np
import numpy.matlib
import os
import h5py


filenames = ['TS_W.mat', 
             'TS_N3.mat', 
             'TS_MCS.mat',
             'TS_UWS.mat',
             'TS_CNT.mat']

directory = "C:/Users/fabio.bauer/Documents/Thesis"

# create an empty dictionary to store the fMRI timeseries for each consciousness state
data_dict = {}

# iterate through each filename and load the corresponding data
for filename in filenames:
    # load the data from the MATLAB file using scipy.io.loadmat()
    file_path = os.path.join(directory, filename)
    mat_data = scipy.loadmat(file_path)

    # discard keys that start with '__'
    keys = [key for key in mat_data.keys() if not key.startswith('__')]

    # assuming there's only one key left, this should be the key for the array we care about
    if len(keys) == 1:
        array_key = keys[0]
    else:
        raise ValueError('More than one array found in MATLAB file')

    # convert the MATLAB array to a numpy array
    np_data = np.array(mat_data[array_key], dtype=object)

    # create an empty list to store the arrays for all subjects
    subject_data = []

    # iterate over the subjects
    for subject in np_data[0]:
        # convert the subject's data to a numpy array and append it to the list
        # since the data for each subject is a 2D array, we don't need to add another dimension
        subject_array = np.array(subject)
        subject_data.append(subject_array)

    # extract the consciousness state from the filename
    consciousness_state = os.path.splitext(filename)[0][3:]

    # store the list of subject arrays directly under the consciousness state key
    data_dict[consciousness_state] = subject_data

#np.savez("TS_data", data_dict=data_dict)


# Saving the dictionary
with h5py.File('TS_data.h5', 'w') as hf:
    for key in data_dict:
        for i, array in enumerate(data_dict[key]):
            hf.create_dataset(f'{key}/{i}', data=array)