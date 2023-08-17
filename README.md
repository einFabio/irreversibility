# irreversibility
Master thesis work on measuring states of consciousness using irreversibility in fMRI and a hopf-bifurcation model

Abstract
Understanding the complex dynamics of brain states and their corresponding levels
of consciousness is a pressing concern in contemporary neuroscience. To unpack
these complexities, this study employed the thermodynamic principle of irreversibil-
ity as an entropy-based metric to parse empirical and simulated functional magnetic
resonance imaging (fMRI) data. In empirical data, the metric effectively discerned
between states of wakefulness and deep sleep, but failed to distinguish minimally
conscious from unresponsive wakefulness conditions. Time series for four different
consciousness states were simulated using a whole-brain Hopf bifurcation model. To
fit these simulations to empirical fMRI observations, the Structural Dissimilarity
Index (DSSIM) was used. Irreversibility clearly distinguished between all investi-
gated states in the simulated data. Further, Structural connectivity, as regulated by
a global coupling parameter in the model profoundly altered irreversibility patterns
and optimised fit. In sum, while the irreversibility metric has demonstrated promise
in discerning the dynamics of consciousness states, it could not discern between
empirical minimally conscious and unresponsive wakefulness conditions, warranting
continued inquiry.
Keywords: Consciousness; fMRI; Irreversibility; Disorders of Consciousnes; Whole-
Brain Modelling; Hopf-Bifurication


TSdata_loadmat.py is a file to convert the empirical fMRI matrices from .mat to general .h5py files. This helps with later use in Numpy.
hopf_tenet_consc_states.py is the file where the empirical data is filtered and irreversibility is measured. Also, here the modeling happens and is equally filtered and irreversibility measured. Finally, DSSIM is calculated. The .mat files needed to run the file are available upon request.
tenet_hopf_tenet_plotting_v4.py takes the output from hopf_tenet_consc_states.py, runs statistical analysis and converts results into plots
