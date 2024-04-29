import numpy as np
import sys
import os
import pickle
from os.path import join as pjoin
import time
from neurotwin.applications.smith.smithpiper import (smithpiper_pipeline, get_archetype_temperatures)

# Define neurotwin directory
neurotwin_source_dir: str = 'neurotwin'
sys.path.append(neurotwin_source_dir)
sys.path.append(os.path.dirname(neurotwin_source_dir))

# Create fullpack with symmetrized GSR, binarized data
# Load data file
data_file = pjoin('data', 'neurotwin_ct_adni_hc_downsampled.npz')

# Load the npz file
data = np.load(data_file)

# Extract the dictionary
data_dict = {key: data[key] for key in data}

n_subjects = len(data_dict)
print("number of subjects", n_subjects)
subject_IDs = list(data_dict.keys())

this_run_name = pjoin('smithpiper', 'data', "Neurotwin_CT_ADNI_HC_"+str(int(time.time())))
savepickle = True
print("this_run_name:", this_run_name)

parameters ={
    'n_steps': 1000000,  # Number of iterations for Metropolis generation of data, must be> 1000000
    'T_nominal': 1,       # nominal system temperature of the model for data generation
    'sparsity': 1.e-1,    # L1 norm for archetype estimation
    'n_processors': 8,
    'estimate_h': False,   # do not estimate h (set to 0)
    'temperatures': np.arange(0.4, 1.5, 0.1),   # np.arange(0.4, 1.125, 0.025)
    'binarize': True,     # only False if data is already binarized
    'symmetrize_labels': True,
    'do_GSR': True,
    'evaluate_criticality_stats': True,
    'compute_link_chi': False
    }

# Define an empty dictionary to store processed data
fullpack = dict()

# Iterate over each subject
for key in data_dict.keys():
    # Load the raw BOLD data
    print("\n... processing ", key)
    ts = data_dict[key]  # Get the time series data for the subject
    if 'neurotwin_ct' in key.lower():
        condition = 'AD'  # Replace 'some_condition' with actual condition
    else:
        condition = 'HC'
    # Call the smithpiper_pipeline function and store the result in the fullpack dictionary
    fullpack[key] = smithpiper_pipeline(key, condition, ts, parameters)

get_archetype_temperatures(fullpack, subject_IDs)  # Call function to process final data

# file_path = 'fullpack_neurotwin_ct_adni_hc_downsampled.pkl'
# # Save the dictionary to a JSON file
# with open(file_path, "wb") as pickle_file:
#     pickle.dump(fullpack, pickle_file)
#
# print("Dictionary saved to:", file_path)
