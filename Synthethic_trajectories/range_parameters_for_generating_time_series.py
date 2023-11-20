"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

Parameters for generating plausible syntethic time-series

@author: giacomo
"""
############# Hyperparameters for creating synthetics #############

############# STEPS are created following the GR law #############
# Set constants for the GR law
a = 5.0
b = 1.0

# range of steps amplitudes
max_nsteps = 4 #Number of steps to put in the time series (MAX) ---------------- 5 before
min_step_size, max_step_size=0.005,0.1 # Maximum and minimum step size (e.g Metres) 
magnitude_steps = np.linspace(min_step_size, max_step_size, 10000)

#n_values = gutenberg_richter_law(magnitude_steps, a, b)
#weights_GR=[i/sum(n_values) for i in (n_values)]

### they can be followed by a logarithmic decay
decay_onA = range(2) # If decay is on (=1), the largest step will also have a logarithmic decay
Adec = 0.02  # Amplitude of the decay  (MAX) ---------------- 0.05
Tdec = 10 # Decay constant  (MAX)

############# SSEs are created with arctangents  #############
num_arctan = 10 #(MAX) number of slow slip events 
max_Aarctan = 0.5 #Maximum amplitude of arctan function 
max_Darctan = 100 #Controls max. width of arctan function (in time. days) 

############# Gaussian shapes #############
num_gau = 5 #(MAX) number of gaussian shapes to put in the signal  
max_Agau = 0.008 #Maximum amplitude of the gaussian function
max_Dgau = 90 #500 # Controls max. width (in time) of the gaussian function ---------------- 500

############# Seasonals #############
seas_freqs = np.array([1,2]) # Oscillations per year (Frequency for periodic functions)
seas_amp = 0.005 #this value is from the max of gratsid fits #amplitude of the seasonal signal  (MAX)CNN ---------------- 0.03
#  0.00066 for E_N
sto_mulA=[0,0,0]+list(range(10)) # controls the stochasticity of the seasonal #before [0,0,0]

############# Offsets and trend #############
mc_on = 1 # boolean to decide whether to include an offset and trend
sec_rateA=np.linspace(-0.00027,0.00027) #0.00027 is 10 cm/yr  # Secular velocity (e.g. Metres per day) --------------- 0.00022 E_N
offsetA = np.linspace(-0.01,0.01,1000) # offset in cm --------------- 0.0001 before

max_gap=5
#noise_level = 5 # Noise level as a fraction of the seasonal amplitude.  (MAX)

### stochastic trend and offset
sec_rate = random.sample(list(sec_rateA), 1)[0] 
offset=random.sample(list(offsetA), 1)[0]
### stochasticity of seasonals
sto_mul = random.sample(list(sto_mulA), 1)[0]
### If decay is on (=1), the largest step will also have a logarithmic decay
decay_on=random.sample(list(decay_onA), 1)[0]

print('Parameters imported')