# %%load packages and define funtions
import scipy.io as scipy
from scipy import signal
from scipy import stats
import numpy as np
from numpy import dot, flip, tile, corrcoef
import numpy.matlib
from skimage import metrics
from copy import deepcopy
import math
import h5py

def tenet(timeseries, delta_t, nonmean):
    #Computing the forward and reversed correlation coefficient matrix
    fc_forward = np.corrcoef(timeseries[:,:-delta_t], timeseries[:,delta_t:])  
    ts_rev = np.flip(timeseries, axis=1)
    fc_reversed = np.corrcoef(ts_rev[:,:-delta_t], ts_rev[:,delta_t:])
    
    # Extract the cross correlation
    m = timeseries.shape[0]
    fc_forward = fc_forward[:m, m:] 
    fc_reversed = fc_reversed[:m, m:] 
    
    #mutual information matrix
    i_tau_forward = -0.5*np.log((1-fc_forward**2))
    i_tau_reversed = -0.5*np.log((1-fc_reversed**2))
    
    #compute squared distance 
    square_dist = np.square(i_tau_forward.flatten() - i_tau_reversed.flatten())
    
    #take the non-reversibility of region pairs of the top quartile
    if nonmean == True:    
        #index top quartile values to increase sensitivity
        index = square_dist > np.quantile(square_dist,0.75)
        
        #compute non-reversibility i
        non_rever = np.mean(square_dist[index])
        
    else:
        non_rever = np.mean(square_dist)
        
    return(non_rever)

def feat_phase(nparcells, bold_nsub, filt_a, filt_b):     
       
        #extracting all features and phases from a 2D timeseries
        #filtering every parcellation
        phases_e = np.zeros((nparcells, np.shape(bold_nsub)[1] ))
        bold_feats = [] #empty list to add filtered bold to 
        for parcnum in np.arange(0, nparcells): #subsetting every parcellation
            bold_nsub[parcnum, :] = bold_nsub[parcnum, :] - np.mean(bold_nsub[parcnum, :]) #substracting the mean activity 
            bold_filt = signal.filtfilt(filt_b, filt_a, bold_nsub[parcnum, :], padtype = 'odd', padlen = 3*(max(len(filt_b),len(filt_a))-1)) #https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function
            bold_feats.append(bold_filt) #bandpass to create a narrowband signal, bc. hilbert extracts the phase of the frequency with most power at each timepoint
            bold_hilb = signal.hilbert(bold_filt - np.mean(bold_filt) )   #substract means across all regions for all timepoints and convert the signal into complex numbers to extract phase(j) and amplitude
            phases_e[parcnum, :] = np.angle(bold_hilb) #returns the phase angle of our hilbertized signal at that point in time as complex argument in radians
        bold_filtered = np.vstack(bold_feats)   #filtered bold signal for one subject 
        
        return phases_e, bold_filtered       
    
def metastability(phases_e):
    
        #computing metastability for every subject   
        phase_complex = -1*np.cos(phases_e) + 1j* np.sin(phases_e) # eulers formular, * by imaginary # gives anticlockwise rotation around origin in complex plain
        phase_abssum = abs(np.sum(phase_complex, axis = 0) ) #sum of all parcellation phases
        phase_byparcell = phase_abssum / nparcells   #average for one parcell                                      
        phase_nan = phase_byparcell[~np.isnan(phase_byparcell)] # getting rid of nans
        meta = np.std(phase_nan, ddof = 1) #metastability std for every subject 
        
        return meta 
    
def get_first_key(dictionary):
    
    for key in dictionary:
        return key
    raise IndexError


def get_smallest_second_dimension_size(arrays):
    
    # Initialize with a large value
    smallest_size = np.inf

    # Iterate over each array in the list
    for arr in arrays:
        # Get the second dimension size of the current array
        current_size = arr.shape[1]

        # Update smallest_size if current_size is smaller
        if current_size < smallest_size:
            smallest_size = current_size

    return smallest_size


# %%Load the empirical data into matrices  and define parameters

hopf_freq = scipy.loadmat('hopf_freq_AAL90_W.mat') #the output of Compute_Hopf_freq_AAL90
f_diff_mat = scipy.loadmat('f_diff.mat')
f_diff = f_diff_mat["f_diff"] #f_diff is intrinsic frequ of nodes
groupings = scipy.loadmat("groupings.mat") #resting state networks for AAL90
int_arr = groupings["grouping"].item() #extract the items from the dict
rsn = int_arr[1][0][2] #relevant 90x6 with 6rsn's of 1/0
rsn = np.array(rsn, dtype = bool) #convert to boolean for indexing

# Loading the timeseries data for each consciousness state in a dictionary
# each consc state has a list of subject arrays with 90 parcellations 
# of x timesteps

data_dict = {}
with h5py.File('TS_data.h5', 'r') as hf:
    for key in hf.keys():
        data_dict[key] = [np.array(hf[f'{key}/{i}']) for i in range(len(hf[key]))]
    
    
#%% Analysing each consciousness state seperately in empirical data

#initializing arrays for metrics to be saved to
tenet_emp_mean_array = np.zeros((len(data_dict.keys()), 1))
tenet_emp_rsn_mean_array = np.zeros((len(data_dict.keys()), 6))
fc_emp_mean_array = np.zeros((len(data_dict.keys()), 90, 90))
fc_emp_eye_array = np.zeros((len(data_dict.keys()), 90, 90)) 
glob_coup_emp_array = np.zeros((len(data_dict.keys()), 90)) 
meta_emp_mean_array = np.zeros((len(data_dict.keys()), 1)) 
cotsampling_emp_list = []
counter_sub_list = []
tenet_emp_list = []

for counter_state, key in enumerate(data_dict.keys()):
    consc_data = data_dict[key]

    nsub = len(consc_data)
    tmax = get_smallest_second_dimension_size(consc_data) # shortest subj timewindow

    #Defining parameters 
    nparcells = 90 #Parcellations via AAL90
    nparcells_vec = np.ones((nparcells,0))*nparcells
    tenet_delta = 1 #timeshift for tenet analysis  it's comparing all elements 
    #in the time series except the last one (timeseries[:,:-1]) with all elements 
    #in the time series except the first one (timeseries[:,1:]).
    indexsub = np.arange(0, nsub) 
    indexparcell = np.arange(0, nparcells)
    t_window = 15
      
    # Cut all arrays to the shortest length along the second dimension (time)
    cut_arrays = [arr[:, :tmax] for arr in consc_data]
    tsdata = np.stack(cut_arrays) #convert to npy array from list of arrays
        
    #computing functional connectivity as a central metric
    fcdata = np.zeros((0, nparcells, nparcells))    
    for i in np.arange(0, nsub):  
        fcfeats = corrcoef(tsdata[i, :, :]) #compute the functional connectivity for each subject 
        fcfeats = np.expand_dims(fcfeats, axis = 0) #expand by a dimension as to have the same dimensions to conc.
        fcdata = np.concatenate((fcdata, fcfeats))     
    
    #We substract the mean from the functional connectivity matrix
    fc_mean = np.squeeze(np.mean(fcdata, 0) ) #squeeze to get rid of nsub axis
    fc_emp = fc_mean - fc_mean * np.eye(nparcells) # mean of FC with diagonal 0
    global_connect_emp = np.mean(fc_emp, 1) #flattened mean FC


    # construct the filter for the signal
    lowp_freq_filt = 0.01 #lowpass frequency of filter
    highp_freq_filt = 0.1 #highpass
    sampl_int = 2 #sampling intervall 2s
    order = 2 #2nd order butterworth filter
    nyq_freq = 1/(2 * sampl_int) #Nyquist frequency
    #butterworth bandpass non-dimensional frequency
    bandpass_freq = np.array([lowp_freq_filt / nyq_freq, highp_freq_filt / nyq_freq])   
    filt_b, filt_a = signal.butter(order, bandpass_freq, btype="bandpass")


    # Zero phase filtering the bold signal for every subject and computing Metastability,
    # functional connectivity and non reversibility/entropy (=tenet function) 
    tsdata_copy = deepcopy(tsdata) #deepcopying to not alter the original in next computation                
    fcdata_filt = np.zeros((nsub,nparcells,nparcells) )     
    bold_filt_ts = np.zeros((nsub,nparcells,tmax))
    meta_emp = np.zeros(nsub)
    tenet_emp = np.zeros((nsub))
    tenet_emp_rsn = np.zeros((len(rsn[0,:]),nsub))
    tenet_emp_rsn_mean = np.zeros((len(rsn[0,:])))
    n = len(np.arange(0, (tmax-t_window*2), t_window))-1 #number of steps in timeseries
    x = n*(n+1)/2 #the sum of the first n natural numbers, where n is steps -1
    cotsampling_emp = np.zeros(int(nsub * x)) 
    
    counter_sub = 0 #subject counter
    
    for sub in indexsub:   
        bold_nsub = np.squeeze(tsdata_copy[sub, :, :])
        phases_e, bold_filtered = feat_phase(nparcells, bold_nsub, filt_a, filt_b) #filtering 
        
        #computing core measurements
        fcdata_filt[sub,:,:] = corrcoef(bold_filtered) #Functional connectivity matrix 
        meta_emp[sub] = metastability(phases_e) 
        tenet_emp[sub] = tenet(bold_filtered, tenet_delta, nonmean=False) #entropy production via tenet
        
        """
        #computing time reversibility for each resting state network
        #Indexing model_filtered parcellations that belong to each of 
        # the 6 resting state network subsets to get their timeseries
        # then perform tenet for every rsn and every subject
        for i in range(len(rsn[0,:])):
            rsn_emp_ts = bold_filtered[rsn[:, i]]
            tenet_rsn = tenet(rsn_emp_ts, tenet_delta, 1)
            tenet_emp_rsn[i,sub] = tenet_rsn        
        """
        
        #computing the distribution of all KOL distances   
        for counter_t0, time0 in enumerate(np.arange(0, (tmax-t_window*2), t_window)):
            corrcoef_t0 = corrcoef(bold_filtered[:,time0:time0+31].T, rowvar = False) #FC in filtered bold in 30s intv
            cc_t0_subdiag = corrcoef_t0[np.triu_indices_from(corrcoef_t0, k = 1)] #lower triangle of the mirrored matrix
            
            for counter_t1, time1 in enumerate(np.arange(0, (tmax-t_window*2), t_window)):
                corrcoef_t1 = corrcoef(bold_filtered[:,time1:time1+t_window*2+1].T, rowvar = False)
                cc_t1_subdiag = corrcoef_t1[np.triu_indices_from(corrcoef_t1, k = 1)]
                cc_of_cc = corrcoef(cc_t0_subdiag, cc_t1_subdiag, rowvar = False) #correlation of connectivity across 31s timesteps
                if counter_t1 > counter_t0:
                    cotsampling_emp[counter_sub] = cc_of_cc[1,0]  
                    counter_sub += 1 #accumulation across all subjects
    
    tenet_emp_mean = np.mean(tenet_emp)
    tenet_emp_rsn_mean = np.mean(tenet_emp_rsn, axis=1)
    fc_emp_mean = np.squeeze(np.mean(fcdata_filt, 0))
    fc_emp_eye = fc_emp_mean - fc_emp_mean * np.eye(nparcells) #substracting the diagonal
    glob_coup_emp = np.mean(fc_emp_eye,1)  #global brain functional connectivity
    meta_emp_mean = np.mean(meta_emp) #mean metastability across subjects
    
    #saving each measure for each consciousness state to plot later
    tenet_emp_list.append(tenet_emp)
    tenet_emp_mean_array[counter_state] = tenet_emp_mean
    tenet_emp_rsn_mean_array[counter_state] = tenet_emp_rsn_mean
    fc_emp_mean_array[counter_state] = fc_emp_mean
    fc_emp_eye_array[counter_state] = fc_emp_eye
    glob_coup_emp_array[counter_state] = glob_coup_emp
    meta_emp_mean_array[counter_state] = meta_emp_mean
    cotsampling_emp_list.append(cotsampling_emp)
    
tenet_emp_array = np.array(tenet_emp_list, dtype = object) #changing type for later saving

# %% Simulating bold data with a hopf model based on DOI:10.1038/s41598-017-03073-5

#fixed model parameters
nsub_sim = 100 #x simulated participants in the hopf model
indexsub_sim = np.arange(0, nsub_sim) 
tmax_sim = 200 #hardcoded timewindow for simulation
trials = 1 # how many trials to run the model for = randomizer
gmax = 4 #coupling G max
step = 0.02 #trying couplings in these steps
steps = int(gmax/step) #trying different coupling paras in steps within a trial
                       #e.g. 10 steps because coupling from 0 to 2 in 0.2 steps
struct_connect = scipy.loadmat("SC.mat") #load structural connectivity, a mean of healthy subjects
struct_connect = struct_connect["SC"] / np.amax(struct_connect["SC"]) * 0.2 # why divide by maximum value? 
omega_positive = tile(2 * np.pi * f_diff.T, (1, 2)) #f_diff is intrinsic frequ of nodes, *pi is radians for complex plain
omega = np.vstack((omega_positive[:,0] * -1, omega_positive[:,0])).T
delta_t = 0.1 * sampl_int / 2 #timestep, why 0.1?

sig = 0.05 #bifurication parameter of hopf model
delta_sig = np.sqrt(delta_t) * sig                    

#arrays for saving the model analysis parameters over many trials
# define the variable names as keys in a dictionary
arrays = {'fc_dist_array': None, 
          'fc_fit_cc_array': None,
          'fcd_fit_ks_array': None, 
          'fc_fit_ssim_array': None,
          'meta_fit_array': None, 
          'fcd_fit_pval_array': None,
          'globcoup_fit_cc_array': None,
          'globcoup_fit__mean_array': None, 
          'tenet_model_mean_array': None}

# create the arrays and assign them to the dictionary values
for name in arrays:
    arrays[name] = np.zeros((trials, steps))

# unpack the dictionary values into separate variables for easy access
(
    fc_dist_array,
    fc_fit_cc_array,
    fcd_fit_ks_array,
    fc_fit_ssim_array,
    meta_fit_array,
    fcd_fit_pval_array,
    globcoup_fit_cc_array,
    globcoup_fit_mean_array,
    tenet_model_mean_array,
) = arrays.values()

cotsampling_simul_array = [] #needs to be a list of varying array sizes
tenet_model_subj = np.zeros((steps,nsub_sim))
tenet_model_array = np.zeros((trials,steps,nsub_sim))
fc_simul_eye_array = np.zeros((steps,nparcells,nparcells))

#separate size arrays for the resting state network analysis
tenet_model_rsn = np.zeros((len(rsn[0,:]),nsub_sim)) 
tenet_model_rsn_array = np.zeros((trials,len(rsn[0,:]),steps))


#running the model 'trial' times 
for n in np.arange(0, trials):
    #arrays for model signal filtering
    model_filt_ts = np.zeros((nsub_sim,nparcells,tmax_sim))   
    model_phases = np.zeros((nparcells,tmax_sim))
    fc_model = np.zeros((nsub_sim,nparcells,nparcells))
    fc_simul_mean = np.zeros((steps,nparcells,nparcells))     
    meta_simul = tenet_model = np.zeros(nsub_sim)
    ts_steps = len(np.arange(0, (tmax-t_window*2), t_window))-1 #number of steps in timeseries
    natural_sum = ts_steps*(ts_steps+1)/2 #the sum of the first n natural numbers, where n is steps -1
    cotsampling_simul = np.zeros(int(nsub_sim * natural_sum))
    
    #arrays for the model analysis
    # define the variable names as keys in a dictionary
    variables = {'fc_dist': None, 
                 'globcoup_fit_cc': None,
                 'globcoup_fit_mean': None, 
                 'fc_fit_cc': None,
                 'fc_fit_ssim': None, 
                 'meta_fit': None,
                 'fcd_fit_ks': None, 
                 'fcd_fit_pval': None,
                 'tenet_model_mean': None}
    
    # create the variables and assign them to the dictionary values
    for name in variables:
        variables[name] = np.zeros(steps)
    
    #unpack the dictionary values into separate variables
    (
        fc_dist,
        globcoup_fit_cc,
        globcoup_fit_mean,
        fc_fit_cc,
        fc_fit_ssim,
        meta_fit,
        fcd_fit_ks,
        fcd_fit_pval,
        tenet_model_mean,
    ) = variables.values()
    
    #separate size array for the resting state network analysis
    tenet_model_rsn_mean = np.zeros((steps,len(rsn[0,:])))

    
    #Hopf model with altering levels of global coupling parameter glob_coupl
    ##
    ###
    #### FOR COUNTER COUPLING
    for counter_coupl, glob_coupl in enumerate(np.arange(step,gmax+step,step)): 
        weigh_sc = glob_coupl * struct_connect #weighing the structural connectivity %wC
        weigh_sc_sum =  np.sum(weigh_sc, 1) #the relative structural connection of one parcell to all others
        weigh_sc_col = tile(weigh_sc_sum, (2,1)).T # to extraction of xj (=columns) for Cij*xj %sumC

        #
        ##
        ### FOR STUDIENTEILNEHMER (15)
        for counter_sub, sub in enumerate(indexsub_sim):
            bifur_a = -0.02 * np.ones((nparcells, 1)) #"a" in equation, stable oscillation a=0
            model_state = 0.1 * np.ones((nparcells, 1)) # z = x+i*y  where x = z[:,0], y = z[:,1]
            model_nsub = np.zeros((tmax_sim, nparcells))
            counter_sampl = 0
            
            #run model 100 times to randomize result
            for t in np.arange(0, 100, delta_t):
                model_fcd = dot(weigh_sc, model_state) - weigh_sc_col * model_state # sum(Cij*xi) - sum(Cij)*xj
                state_flip = flip(model_state, 1) # because (x*x + y*y) %zz ???
                model_det = model_state + delta_t * (bifur_a * model_state + state_flip * omega 
                                                  - model_state*(
                                                      np.square(model_state) + np.square(state_flip) )
                                                  + model_fcd)
                noise = delta_sig * np.random.randn(nparcells, 1) #scaled gaussian noise to add
                model_state = model_det + noise 
            
            #model new signal for each sub based on mean weighted structural connectivity
            for t in np.arange(0,(tmax_sim-1) * sampl_int, delta_t):
                model_fcd = dot(weigh_sc, model_state) - weigh_sc_col * model_state # sum(Cij*xi) - sum(Cij)*xj
                state_flip = flip(model_state, 1) # because (x*x + y*y) %zz ???
                model_det = model_state + delta_t * (bifur_a * model_state + state_flip * omega 
                                                  - model_state*(
                                                      np.square(model_state) + np.square(state_flip) )
                                                  + model_fcd)
                noise = delta_sig * np.random.randn(nparcells, 1) #scaled gaussian noise to add
                model_state = model_det + noise 
                
                #align model and empirical timesteps by sampling when they match
                if math.fmod(t, sampl_int) < 0.01:    
                    counter_sampl += 1           
                    model_nsub[counter_sampl, :] = model_state[:,0].conj().T #complex conjugate transpose
                    
            model_nsub = model_nsub.conj().T  
            model_phases, model_filtered = feat_phase(nparcells, model_nsub, filt_a, filt_b) 
            
            print(sub)
            
            #computing core measures
            fc_model[sub, :, :] = corrcoef(model_filtered) 
            meta_simul[sub] = metastability(model_phases)
            tenet_model[sub] = tenet(model_filtered, tenet_delta,0) #entropy production via tenet
            
            """
            #computing time reversibility for each resting state network
            # Indexing model_filtered parcellations that belong to each of 
            # the 6 resting state network subsets to get their timeseries            
            for i in range(len(rsn[0,:])):
                rsn_model_ts = model_filtered[rsn[:, i]]
                tenet_model_rsn[i,sub] = tenet(rsn_model_ts, tenet_delta)   
            """
            #computing the distribution of all kolgomorov distances   

            for counter_t0, time0 in enumerate(np.arange(0, (tmax_sim-t_window*2), t_window)):
                corrcoef_t0 = corrcoef(model_filtered[:,time0:time0+t_window+1].T, rowvar = False) 
                cc_t0_subdiag = corrcoef_t0[np.triu_indices_from(corrcoef_t0, k = 1)] 
                            
                for counter_t1, time1 in enumerate(np.arange(0, (tmax_sim-t_window*2), t_window)):
                    corrcoef_t1 = corrcoef(model_filtered[:,time1:time1+t_window+1].T, rowvar = False)
                    cc_t1_subdiag = corrcoef_t1[np.triu_indices_from(corrcoef_t1, k = 1)]                            
                    cc_of_cc = corrcoef(cc_t0_subdiag, cc_t1_subdiag, rowvar = False) 
                    
                    if counter_t1 > counter_t0:
                        cotsampling_simul[counter_sub] = cc_of_cc[1,0]  
                        
        cotsampling_simul_array.append(cotsampling_simul)   
                        
        #Core model metrics for each coupling
        fc_simul_mean[counter_coupl] = np.squeeze(np.mean(fc_model,0)) #mean of subjects
        fc_simul_eye = fc_simul_mean - fc_simul_mean * np.eye(nparcells) #substracting the diagonal
        #glob_coup_simul = np.mean(fc_simul_eye, 1)  #global brain functional connectivity
        meta_sim_mean = np.mean(meta_simul) #mean metastability across subjects
        
        print("coupl " + str(counter_coupl))
        
        #compute tenet mean for each model trial and coupling
        tenet_model_mean[counter_coupl] = np.mean(tenet_model) #mean across all subjects
        tenet_model_subj[counter_coupl] = tenet_model
        tenet_model_rsn_mean[counter_coupl] = np.mean(tenet_model_rsn, 1) #mean across all subj for each rsn
        
        """
        #Model analysis via fitting metrics for each coupling parameter value
        fc_simul_subdiag = fc_simul_mean[counter_coupl][np.triu_indices_from(fc_simul_mean[counter_coupl], k = 1)]
        fc_emp_subdiag = fc_emp_mean[np.triu_indices_from(fc_emp_mean, k = 1)]  
        fc_diff_sqrd = np.power((fc_simul_subdiag - fc_emp_subdiag), 2)
        fc_dist[counter_coupl] = np.mean(fc_diff_sqrd) 
        
        fc_fitting = corrcoef(np.arctanh(fc_simul_subdiag),np.arctanh(fc_emp_subdiag))
        fc_fit_cc[counter_coupl] = fc_fitting[1,0] #Functional connectivity fitting
        
        fc_fit_ssim[counter_coupl] = metrics.structural_similarity(fc_simul_mean[counter_coupl], fc_emp_mean)
    
        globcoup_fit_cc[counter_coupl] = corrcoef(glob_coup_emp, glob_coup_simul)[0,1]
        globcoup_fit_sqrd = np.power((glob_coup_emp - glob_coup_simul), 2)
        globcoup_fit_mean[counter_coupl] = np.sqrt(np.mean(globcoup_fit_sqrd)) #Global coupling fitting
    
        meta_fit[counter_coupl] = abs((meta_sim_mean) - meta_emp_mean) #Metastability fitting 

        """ 
    
    #saving stats in arrays for each coupling
    tenet_model_mean_array[n] = tenet_model_mean
    tenet_model_array[n] = tenet_model_subj
    tenet_model_rsn_array[n] = tenet_model_rsn_mean.T
    #fc_simul_eye_array[n] = fc_simul_eye
    fc_dist_array[n] = fc_dist 
    fc_fit_cc_array[n] = fc_fit_cc 
    #[n] = fcd_fit_ks 
    fcd_fit_pval_array[n] = fcd_fit_pval 
    #tenet_model_array = np.flip(tenet_model_array)
    globcoup_fit_cc_array[n] = globcoup_fit_cc 
    globcoup_fit_mean_array[n] = globcoup_fit_mean
    meta_fit_array[n] = meta_fit

# %%Computing fitting metrics for each state of consciousness and each
#    coupling parameter. Focus on structural dissimilarity
"""
fc_fit_ssim_array = np.zeros((len(data_dict.keys()), steps))
fc_fit_ssim_coupling = np.zeros((steps))

fcd_fit_ks = np.zeros((steps, 1))
fcd_fit_pval = np.zeros((steps, 1))
fcd_fit_ks_array = np.zeros((len(data_dict.keys()), steps, 1))
fcd_fit_pval_array = np.zeros((len(data_dict.keys()), steps, 1))

for counter_state, consc_state in enumerate(data_dict.keys()):
    fc_emp_mean_state = fc_emp_mean_array[counter_state]
    
    for counter_coupl in np.arange(0,steps):
        ssim_val = metrics.structural_similarity(fc_simul_mean[counter_coupl], fc_emp_mean_state)
        fc_fit_ssim = min(1, 1 - ssim_val) #dissimilarity 
        fc_fit_ssim_coupling[counter_coupl] = fc_fit_ssim
        
        #Functional connectivity dynamics fitting via max distance between both distribution graphs, kolgomorov distance
        fcd_fit_ks[counter_coupl], fcd_fit_pval[counter_coupl] = stats.ks_2samp(cotsampling_emp_list[counter_state], cotsampling_simul_array[counter_coupl])
    
    fc_fit_ssim_array[counter_state] = fc_fit_ssim_coupling   
    fcd_fit_ks_array[counter_state] = fcd_fit_ks
    fcd_fit_pval_array[counter_state] = fcd_fit_pval
       
    

#save into arrays, maybe redundant if original array can fit it???
    fc_dist_array[n] = fc_dist #
    fc_fit_cc_array[n] = fc_fit_cc #
    fc_fit_ssim_array[n] = fc_fit_ssim #
    
    fcd_fit_ks_array[n] = fcd_fit_ks #
    fcd_fit_pval_array[n] = fcd_fit_pval #
    
    globcoup_fit_cc_array[n] = globcoup_fit_cc #
    globcoup_fit_mean_array[n] = globcoup_fit_mean #
    
    meta_fit_array[n] = meta_fit #
"""  

# %%Computing fitting metrics for each state of consciousness and each
#    coupling parameter. Focus on structural dissimilarity


##################
##################
##################

fc_fit_ssim_array = np.zeros((len(data_dict.keys()), steps))
fc_fit_ssim_coupling = np.zeros((steps))

for counter_state, consc_state in enumerate(data_dict.keys()):
    fc_emp_eye_state = fc_emp_eye_array[counter_state]
    
    for counter_coupl in np.arange(0,steps):
        ssim_val = metrics.structural_similarity(fc_simul_eye[counter_coupl], fc_emp_eye_state)
        fc_fit_ssim = min(1, 1 - ssim_val) #dissimilarity 
        fc_fit_ssim_coupling[counter_coupl] = fc_fit_ssim
        
    fc_fit_ssim_array[counter_state] = fc_fit_ssim_coupling 
# %%saving
# save the arrays to .npz files

np.save("ts_n3_filtered_sub1", bold_filtered)
np.savez("tenet_emp_model_data", tenet_emp_array = tenet_emp_array,  
         tenet_model_array = tenet_model_array)
np.savez("tenet_fit_stats.npz", fc_fit_ssim_array=fc_fit_ssim_array,
         fcd_fit_ks_array=fcd_fit_ks_array,
         fcd_fit_pval_array = fcd_fit_pval_array)
np.savez("fc_emp_model.npz", fc_simul_mean = fc_simul_mean, fc_emp_mean_array = fc_emp_mean_array)
np.savez("structural_connectivity.npz", struct_connect = struct_connect)    