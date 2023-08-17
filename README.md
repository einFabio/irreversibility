# irreversibility
Master thesis work on measuring states of consciousness using irreversibility in fMRI and a hopf-bifurcation model


TSdata_loadmat.py is a file to convert the empirical fMRI matrices from .mat to general .h5py files. This helps with later use in Numpy.
hopf_tenet_consc_states.py is the file where the empirical data is filtered and irreversibility is measured. Also, here the modeling happens and is equally filtered and irreversibility measured. Finally, DSSIM is calculated. The .mat files needed to run the file are available upon request.
tenet_hopf_tenet_plotting_v4.py takes the output from hopf_tenet_consc_states.py, runs statistical analysis and converts results into plots
