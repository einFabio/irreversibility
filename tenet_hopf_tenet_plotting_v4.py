import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import itertools
from matplotlib.ticker import FuncFormatter
#from scipy.stats import mannwhitneyu

def scientific_single(exponent):
    def formatter(value, _):
        # Format tick values to scientific notation
        # A single multiplier at the top of the axis
        scaled_value = value * 10**exponent
        return f"{scaled_value:.0f}"
    return formatter

# Set global font size
plt.rcParams['font.size'] = 12

#%% Loading data

data_fit = np.load("tenet_fit_stats.npz", allow_pickle=True)
data = np.load("tenet_emp_model_data.npz", allow_pickle=True)


fc_fit_ssim_array = data_fit["fc_fit_ssim_array"]
tenet_model_array = data["tenet_model_array"]
tenet_emp_array = data["tenet_emp_array"]
#tenet_emp_mean_array = data["tenet_emp_mean_array"]

# Define the consciousness state labels
consciousness_states = ['CNT', 'MCS', 'N3', 'UWS', 'W']

# Define coupling G
step = 0.02
gmax = 4

# Separate the datasets based on consciousness states
wakefulness_states = ['W', 'N3']
disorders_states = ['CNT', 'MCS', 'UWS']

wakefulness_indices = [consciousness_states.index(state) for state in wakefulness_states]
disorders_indices = [consciousness_states.index(state) for state in disorders_states]

wakefulness_tenet_emp_array = tenet_emp_array[wakefulness_indices]
disorders_tenet_emp_array = tenet_emp_array[disorders_indices]

wakefulness_fc_fit_ssim_array = fc_fit_ssim_array[wakefulness_indices]
disorders_fc_fit_ssim_array = fc_fit_ssim_array[disorders_indices]


#%% Analysis

# Compute the median values for each consciousness state
wakefulness_median_values = [np.median(emp_values) for emp_values in wakefulness_tenet_emp_array]
disorders_median_values = [np.median(emp_values) for emp_values in disorders_tenet_emp_array]

# Sort the wakefulness and disorders of consciousness states based on median values in descending order
wakefulness_sorted_states = [state for _, state in sorted(zip(wakefulness_median_values, wakefulness_states), reverse=True)]
disorders_sorted_states = [state for _, state in sorted(zip(disorders_median_values, disorders_states), reverse=True)]

# Sort the wakefulness and disorders of consciousness tenet_emp_array based on the sorted states
wakefulness_sorted_emp_array = [wakefulness_tenet_emp_array[wakefulness_states.index(state)] for state in wakefulness_sorted_states]
disorders_sorted_emp_array = [disorders_tenet_emp_array[disorders_states.index(state)] for state in disorders_sorted_states]

# Compute the overall minimum and maximum values for setting y-axis limits
wakefulness_ymin = np.min([np.min(emp_values) for emp_values in wakefulness_sorted_emp_array])
wakefulness_ymax = np.max([np.max(emp_values) for emp_values in wakefulness_sorted_emp_array])
wakefulness_y_range = wakefulness_ymax - wakefulness_ymin

disorders_ymin = np.min([np.min(emp_values) for emp_values in disorders_sorted_emp_array])
disorders_ymax = np.max([np.max(emp_values) for emp_values in disorders_sorted_emp_array])
disorders_y_range = disorders_ymax - disorders_ymin

# Initialize the lists to store the p-values
wakefulness_p_values = []
disorders_p_values = []

# Perform the Wilcoxon rank-sum test for every pair of wakefulness states
for i, j in itertools.combinations(range(len(wakefulness_sorted_emp_array)), 2):
    stat, p = ranksums(wakefulness_sorted_emp_array[i], wakefulness_sorted_emp_array[j])
    wakefulness_p_values.append((i, j, p))
    print(f"Wakefulness states {wakefulness_sorted_states[i]} and {wakefulness_sorted_states[j]}: stat = {stat:.5f}, p = {p:.5f}")

# Perform the Wilcoxon rank-sum test for every pair of disorders states
for i, j in itertools.combinations(range(len(disorders_sorted_emp_array)), 2):
    stat, p = ranksums(disorders_sorted_emp_array[i], disorders_sorted_emp_array[j])
    disorders_p_values.append((i, j, p))
    print(f"Disorders states {disorders_sorted_states[i]} and {disorders_sorted_states[j]}: stat = {stat:.5f}, p = {p:.5f}")
    

#%% Control dataset

# Define the consciousness state labels
wakefulness_states = ['W', 'N3']

# Compute the median values for each consciousness state
wakefulness_median_values = [np.median(emp_values) for emp_values in wakefulness_tenet_emp_array]

# Sort the control consciousness states based on median values in descending order
wakefulness_sorted_states = [state for _, state in sorted(zip(wakefulness_median_values, wakefulness_states), reverse=True)]

# Sort the control consciousness states tenet_emp_array based on the sorted states
wakefulness_sorted_tenet_emp_array = [wakefulness_tenet_emp_array[wakefulness_states.index(state)] for state in wakefulness_sorted_states]

# Compute the overall minimum and maximum values for setting y-axis limits
wakefulness_ymin = np.min([np.min(emp_values) for emp_values in wakefulness_sorted_tenet_emp_array])
wakefulness_ymax = np.max([np.max(emp_values) for emp_values in wakefulness_sorted_tenet_emp_array])
wakefulness_y_range = wakefulness_ymax - wakefulness_ymin

# Initialize the list to store the p-values
wakefulness_p_values = []

# Perform the Wilcoxon rank-sum test for every pair of control consciousness states
for i, j in itertools.combinations(range(len(wakefulness_sorted_emp_array)), 2):
    stat, p = ranksums(wakefulness_sorted_emp_array[i], wakefulness_sorted_emp_array[j])
    wakefulness_p_values.append((i, j, p))
    print(f"wakefulness states {wakefulness_sorted_states[i]} and {wakefulness_sorted_states[j]}: stat = {stat:.5f}, p = {p:.5f}")
    
    
# Create a figure for the box plots of disorders of consciousness states
plt.figure(figsize=(10, 6))

# Place all box plots for disorders in the same subplot
parts = plt.boxplot(wakefulness_sorted_emp_array, patch_artist=True)

# Change the color and opacity of the boxes
for box in parts['boxes']:
    box.set_facecolor('blue')
    box.set_alpha(0.5)

#plt.title('Wakefulness and Sleep')
plt.ylabel('Irreversibility log()')  # Add multiplier to the y-axis label
plt.xticks(np.arange(1, len(wakefulness_sorted_states) + 1), wakefulness_sorted_states, rotation=90)
#plt.ylim(wakefulness_ymin - 0.1 * wakefulness_y_range, wakefulness_ymax + 0.1 * wakefulness_y_range)  # Increase ylim to lower the bars
plt.yscale('log')

exponent = 3

# Apply the scientific formatter to the y-axis
formatter = FuncFormatter(scientific_single(exponent))
plt.gca().yaxis.set_major_formatter(formatter)


# Initialize y coordinate for the next bar
next_y = wakefulness_ymax

# Add a marker for each pair of states
for i, j, p_value in sorted(wakefulness_p_values, key=lambda x: (x[0], -x[1])):
    if p_value <0.05:
        y = next_y + 0.05 * wakefulness_y_range
        next_y = y  # Update next_y
        plt.plot([i+1, j+1], [y, y], color='black')
        plt.text((i+j+2)/2, y, "***", ha='center', va='bottom', color="black")
        

plt.tight_layout()

# Save the plot in high resolution
plt.savefig('irreversibility_control_emp.png', dpi=300)

plt.show()

    
#%% Disorders of Consciousness dataset

# Define the consciousness state labels
disorders_states = ['CNT', 'MCS', 'UWS']

# Compute the median values for each consciousness state
disorders_median_values = [np.median(emp_values) for emp_values in disorders_tenet_emp_array]

# Sort the disorders of consciousness states based on median values in descending order
disorders_sorted_states = [state for _, state in sorted(zip(disorders_median_values, disorders_states), reverse=True)]

# Sort the disorders of consciousness tenet_model_array based on the sorted states
disorders_sorted_model_array = [disorders_tenet_emp_array[disorders_states.index(state)] for state in disorders_sorted_states]

# Compute the overall minimum and maximum values for setting y-axis limits
disorders_ymin = np.min([np.min(model_values) for model_values in disorders_sorted_model_array])
disorders_ymax = np.max([np.max(model_values) for model_values in disorders_sorted_model_array])
disorders_y_range = disorders_ymax - disorders_ymin

# Initialize the list to store the p-values
disorders_p_values = []

# Perform the Wilcoxon rank-sum test for every pair of disorders states
for i, j in itertools.combinations(range(len(disorders_sorted_model_array)), 2):
    stat, p = ranksums(disorders_sorted_model_array[i], disorders_sorted_model_array[j])
    disorders_p_values.append((i, j, p))
    print(f"Disorders states {disorders_sorted_states[i]} and {disorders_sorted_states[j]}: stat = {stat:.5f}, p = {p:.5f}")

# Create a figure for the box plots of disorders of consciousness states
plt.figure(figsize=(10, 6))

# Place all box plots for disorders in the same subplot
parts = plt.boxplot(disorders_sorted_model_array, patch_artist=True)

# Change the color and opacity of the boxes
for box in parts['boxes']:
    box.set_facecolor('blue')
    box.set_alpha(0.5)

#plt.title('Disorders of Consciousness States')
plt.ylabel('Irreversibility ($x10^{-4}$)')  # Add multiplier to the y-axis label
plt.xticks(np.arange(1, len(disorders_sorted_states) + 1), disorders_sorted_states, rotation=90)
plt.ylim(disorders_ymin - 0.1 * disorders_y_range, disorders_ymax + 0.2 * disorders_y_range)  # Increase ylim to lower the bars

exponent = 4

# Apply the scientific formatter to the y-axis
formatter = FuncFormatter(scientific_single(exponent))
plt.gca().yaxis.set_major_formatter(formatter)

# Initialize y coordinate for the next bar
next_y = disorders_ymax

# Add a marker for each pair of states
for i, j, p_value in sorted(disorders_p_values, key=lambda x: (x[0], -x[1])):
    if p_value < 0.05:
        y = next_y + 0.05 * disorders_y_range
        next_y = y  # Update next_y
        plt.plot([i+1, j+1], [y, y], color='black')
        plt.text((i+j+2)/2, y, "***", ha='center', va='bottom', color="black") 
    else: None     

plt.tight_layout()

# Save the plot in high resolution
plt.savefig('irreversibility_disorders_emp.png', dpi=300)

plt.show()


#%% Coupling value G with lowest dissimilarity for each consciousness state (Wakefulness)

wakefulness_fc_fit_ssim_mean = np.mean(fc_fit_ssim_array[wakefulness_indices], axis=1)
wakefulness_fc_fit_ssim_min = np.amin(fc_fit_ssim_array[wakefulness_indices], axis=1)
wakefulness_fc_fit_ssim_max = np.amax(fc_fit_ssim_array[wakefulness_indices], axis=1)
wakefulness_min_index = np.argmin(fc_fit_ssim_array[wakefulness_indices], axis=1)
wakefulness_min_dssim_coupling = (np.ones(wakefulness_min_index.shape) + wakefulness_min_index) * 0.02

for n in range(len(wakefulness_indices)):
    print(wakefulness_states[n] + " Mean: " + str("%.5f" % wakefulness_fc_fit_ssim_mean[n]) +
          " Min: " + str("%.5f" % wakefulness_fc_fit_ssim_min[n]) +
          " Max: " + str("%.5f" % wakefulness_fc_fit_ssim_max[n]) +
          " Optimal G: " + str("%.2f" % wakefulness_min_dssim_coupling[n]))

# Coupling value G with lowest dissimilarity for each consciousness state (Disorders)

disorders_fc_fit_ssim_mean = np.mean(fc_fit_ssim_array[disorders_indices], axis=1)
disorders_fc_fit_ssim_min = np.amin(fc_fit_ssim_array[disorders_indices], axis=1)
disorders_fc_fit_ssim_max = np.amax(fc_fit_ssim_array[disorders_indices], axis=1)
disorders_min_index = np.argmin(fc_fit_ssim_array[disorders_indices], axis=1)
disorders_min_dssim_coupling = (np.ones(disorders_min_index.shape) + disorders_min_index) * 0.02

for n in range(len(disorders_indices)):
    print(disorders_states[n] + " Mean: " + str("%.5f" % disorders_fc_fit_ssim_mean[n]) +
          " Min: " + str("%.5f" % disorders_fc_fit_ssim_min[n]) +
          " Max: " + str("%.5f" % disorders_fc_fit_ssim_max[n]) +
          " Optimal G: " + str("%.2f" % disorders_min_dssim_coupling[n]))
    
#%% Boxplot States of Consciousness Model all

# Separate the datasets based on consciousness states
wakefulness_states = ['W', 'N3']
disorders_states = ['CNT', 'MCS', 'UWS']

wakefulness_tenet_model_array = tenet_model_array[0][wakefulness_min_index]
disorders_tenet_model_array = tenet_model_array[0][disorders_min_index]

exponent = 4  # Define the exponent for the y-axis label

# Perform Wilcoxon rank-sum test for each pair of states
all_states = wakefulness_states + disorders_states
all_model_array = list(wakefulness_tenet_model_array) + list(disorders_tenet_model_array)

all_p_values = []
for i, j in itertools.combinations(range(len(all_model_array)), 2):
    stat, p = ranksums(all_model_array[i], all_model_array[j])
    all_p_values.append((i, j, p))
    print(f"{all_states[i]} and {all_states[j]}: stat = {stat:.5f}, p = {p:.5f}")

# Compute the overall minimum and maximum values for setting y-axis limits
all_ymin = np.min([np.min(values) for values in all_model_array])
all_ymax = np.max([np.max(values) for values in all_model_array])
all_y_range = all_ymax - all_ymin

# Create a figure for the box plots of all consciousness states
plt.figure(figsize=(12, 6))

# Place all box plots for all states in the same subplot
parts = plt.boxplot(all_model_array, patch_artist=True)

# Change the color and opacity of the boxes
for box in parts['boxes']:
    box.set_facecolor('blue')
    box.set_alpha(0.5)

#plt.title('Irreversibility of whole brain model at optimal working point for each consciousness state')
plt.ylabel(f'Irreversibility ($x10^{{-{exponent}}}$)')  # Add multiplier to the y-axis label
plt.xticks(np.arange(1, len(all_states) + 1), all_states, rotation=90)
plt.ylim(all_ymin - 0.2 * all_y_range, all_ymax + 0.5 * all_y_range)  # Increase ylim to lower the bars

# Apply the scientific formatter to the y-axis
formatter = FuncFormatter(scientific_single(exponent))
plt.gca().yaxis.set_major_formatter(formatter)

# Initialize y coordinate for the next bar
next_y = all_ymax
"""
# Add a marker for each pair of states
for i, j, p_value in sorted(all_p_values, key=lambda x: (x[0], -x[1])):
    y = next_y + 0.05 * all_y_range
    next_y = y  # Update next_y
    plt.plot([i+1, j+1], [y, y], color='black')
    color = 'red' if p_value < 0.05 else 'black'
    plt.text((i+j+2)/2, y, "***", ha='center', va='bottom', color=color)
"""
plt.tight_layout()
# Save the plot in high resolution
plt.savefig('irreversibility_model.png', dpi=300)
plt.show()

#%% Boxplot States of Consciousness Model Sleep Control

# Separate the datasets based on consciousness states
wakefulness_states = ['W', 'N3']

wakefulness_tenet_model_array = tenet_model_array[0][wakefulness_min_index]

exponent = 3  # Define the exponent for the y-axis label

# Perform Wilcoxon rank-sum test for each pair of states
all_states = wakefulness_states 
all_model_array = list(wakefulness_tenet_model_array) 

all_p_values = []
for i, j in itertools.combinations(range(len(all_model_array)), 2):
    stat, p = ranksums(all_model_array[i], all_model_array[j])
    all_p_values.append((i, j, p))
    print(f"{all_states[i]} and {all_states[j]}: stat = {stat:.5f}, p = {p:.5f}")

# Compute the overall minimum and maximum values for setting y-axis limits
all_ymin = np.min([np.min(values) for values in all_model_array])
all_ymax = np.max([np.max(values) for values in all_model_array])
all_y_range = all_ymax - all_ymin

# Create a figure for the box plots of all consciousness states
plt.figure(figsize=(10, 6))

# Define positions for each boxplot
positions = [1,1.5]

# Place all box plots for all states in the same subplot
parts = plt.boxplot(all_model_array, patch_artist=True, positions=positions)


# Change the color and opacity of the boxes
for box in parts['boxes']:
    box.set_facecolor('blue')
    box.set_alpha(0.5)

#plt.title('Irreversibility of whole brain model at optimal working point for wakefulness and sleep')
plt.ylabel(f'Irreversibility ($x10^{{-{exponent}}}$)')  # Add multiplier to the y-axis label
plt.xticks(np.arange(1, len(all_states) + 1), all_states, rotation=90)
plt.ylim(all_ymin - 0.2 * all_y_range, all_ymax + 0.5 * all_y_range)  # Increase ylim to lower the bars

# Apply the scientific formatter to the y-axis
formatter = FuncFormatter(scientific_single(exponent))
plt.gca().yaxis.set_major_formatter(formatter)

# Adjust x-tick positions and labels to match the boxplot positions
plt.xticks(positions, all_states)

# Set tighter x-axis limits to reduce whitespace on the sides
plt.xlim(0.5, 2)

# Initialize y coordinate for the next bar
next_y = all_ymax

# Adjusted positions for significance bars
bar_positions = [positions[i] for i in [0, 1]]

# Add a marker for each pair of states
for (i, j, p_value) in sorted(all_p_values, key=lambda x: (x[0], -x[1])):
    if p_value < 0.05:
        y = next_y + 0.05 * all_y_range
        next_y = y  # Update next_y
        plt.plot(bar_positions, [y, y], color='black')
        plt.text(np.mean(bar_positions), y, "***", ha='center', va='bottom', color="black")

# Save the plot in high resolution
plt.savefig('irreversibility_model_control.png', dpi=300)
plt.show()

#%% Boxplot States of Consciousness Model DOC

# Separate the datasets based on consciousness states
disorders_states = ['CNT', 'MCS', 'UWS']
disorders_tenet_model_array = tenet_model_array[0][disorders_min_index]

exponent = 3  # Define the exponent for the y-axis label

# Perform Wilcoxon rank-sum test for each pair of states
all_states = disorders_states
all_model_array = list(disorders_tenet_model_array)

all_p_values = []
for i, j in itertools.combinations(range(len(all_model_array)), 2):
    stat, p = ranksums(all_model_array[i], all_model_array[j])
    all_p_values.append((i, j, p))
    print(f"{all_states[i]} and {all_states[j]}: stat = {stat:.5f}, p = {p:.5f}")

# Compute the overall minimum and maximum values for setting y-axis limits
all_ymin = np.min([np.min(values) for values in all_model_array])
all_ymax = np.max([np.max(values) for values in all_model_array])
all_y_range = all_ymax - all_ymin

# Create a figure for the box plots of all consciousness states
plt.figure(figsize=(12, 6))

# Place all box plots for all states in the same subplot
parts = plt.boxplot(all_model_array, patch_artist=True)

# Change the color and opacity of the boxes
for box in parts['boxes']:
    box.set_facecolor('blue')
    box.set_alpha(0.5)

#plt.title('Irreversibility of whole brain model at optimal working point for disorders of consciousness')
plt.ylabel(f'Irreversibility ($x10^{{-{exponent}}}$)')  # Add multiplier to the y-axis label
plt.xticks(np.arange(1, len(all_states) + 1), all_states, rotation=90)
plt.ylim(all_ymin - 0.2 * all_y_range, all_ymax + 0.5 * all_y_range)  # Increase ylim to lower the bars

# Apply the scientific formatter to the y-axis
formatter = FuncFormatter(scientific_single(exponent))
plt.gca().yaxis.set_major_formatter(formatter)

# Initialize y coordinate for the next bar
next_y = all_ymax

# Add a marker for each pair of states
for i, j, p_value in sorted(all_p_values, key=lambda x: (x[0], -x[1])):
    if p_value < 0.05:
        y = next_y + 0.08 * all_y_range
        next_y = y  # Update next_y
        plt.plot([i+1, j+1], [y, y], color='black')
        plt.text((i+j+2)/2, y, "***", ha='center', va='bottom', color="black") 
 
plt.tight_layout()
# Save the plot in high resolution
plt.savefig('irreversibility_model_doc.png', dpi=300)
plt.show()

#%% Line Plot

# Compute the mean and range for each coupling parameter
mean_values = np.mean(tenet_model_array[0], axis=1)
min_values = np.min(tenet_model_array[0], axis=1)
max_values = np.max(tenet_model_array[0], axis=1)

# Compute the mean and range for each coupling parameter
mean_values = np.mean(tenet_model_array[0], axis=1)
std_values = np.std(tenet_model_array[0], axis=1)

# Create the x-axis values based on the coupling parameter range
coupling_parameter = np.arange(step,gmax+step,step) # Range from 0 to 4 with 0.2 increments
x = coupling_parameter[:mean_values.shape[0]]

# Stack empirical tenet data with its optimal DSSIM value
means = [sum(lst) / len(lst) for lst in wakefulness_tenet_emp_array]
ssim_tenet_emp = np.column_stack((means, wakefulness_min_dssim_coupling))

exponent = 3  # Define the exponent for the y-axis label

# Plot the mean line
plt.plot(x, mean_values, color='blue', label='Mean')

# Plot the band of one standard deviation
plt.fill_between(x, mean_values - std_values, mean_values + std_values, color='blue', alpha=0.2, label='Standard Deviation')

# Plot the range of values as shaded regions
plt.fill_between(x, min_values, max_values, color='orange', alpha=0.2, label='Range')

# Plot the dot markers for tenet_emp_mean_array
x_emp = x[:tenet_emp_array.shape[0]]
plt.scatter(ssim_tenet_emp[:, 1], ssim_tenet_emp[:, 0], color='red', marker='o', label='Empirical')

"""
# Add error bars representing the range of values for each point
for i in range(len(wakefulness_states)):
    y = tenet_emp_array[i]
    plt.errorbar(ssim_tenet_emp[i, 1], ssim_tenet_emp[i, 0], yerr=np.std(y), alpha=0.4, color='grey')
"""

# Add labels and title
plt.xlabel('Coupling parameter', fontsize = 10)
plt.ylabel('Irreversibility ($x10^{-3}$)', fontsize=10)
#plt.title('Irreversibility by coupling of modelled data,\n including empirical data at optimal GoF')

# Apply the scientific formatter to the y-axis
formatter = FuncFormatter(scientific_single(exponent))
plt.gca().yaxis.set_major_formatter(formatter)

# Add a legend
plt.legend(loc='best', fontsize=8)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add descriptors next to each dot
for i in range(len(wakefulness_states)):
    plt.text(ssim_tenet_emp[i, 1], ssim_tenet_emp[i, 0], wakefulness_states[i], ha='right', va='top', fontsize =10)
    
# Save the plot in high resolution
plt.savefig('irreversibility_model_emp.png', dpi=300)

# Display the plot
plt.show()
#%% Line Plot

# Compute the mean and range for each coupling parameter
mean_values = np.mean(tenet_model_array[0], axis=1)
min_values = np.min(tenet_model_array[0], axis=1)
max_values = np.max(tenet_model_array[0], axis=1)

# Compute the mean and range for each coupling parameter
mean_values = np.mean(tenet_model_array[0], axis=1)
std_values = np.std(tenet_model_array[0], axis=1)

# Create the x-axis values based on the coupling parameter range
coupling_parameter = np.arange(step,gmax+step,step) # Range from 0 to 4 with 0.2 increments
x = coupling_parameter[:mean_values.shape[0]]

# Stack empirical tenet data with its optimal DSSIM value
means = [sum(lst) / len(lst) for lst in disorders_tenet_emp_array]
ssim_tenet_emp = np.column_stack((means, disorders_min_dssim_coupling))

exponent = 3  # Define the exponent for the y-axis label

# Plot the mean line
plt.plot(x, mean_values, color='blue', label='Mean')

# Plot the band of one standard deviation
plt.fill_between(x, mean_values - std_values, mean_values + std_values, color='blue', alpha=0.2, label='Standard Deviation')

# Plot the range of values as shaded regions
plt.fill_between(x, min_values, max_values, color='orange', alpha=0.2, label='Range')

# Plot the dot markers for tenet_emp_mean_array
x_emp = x[:tenet_emp_array.shape[0]]
plt.scatter(ssim_tenet_emp[:, 1], ssim_tenet_emp[:, 0], color='red', marker='o', label='Empirical')

"""
# Add error bars representing the range of values for each point
for i in range(len(disorders_states)):
    y = tenet_emp_array[i]
    plt.errorbar(ssim_tenet_emp[i, 1], ssim_tenet_emp[i, 0], yerr=np.std(y), alpha=0.4, color='grey')
"""
# Add labels and title
plt.xlabel('Coupling parameter', fontsize = 10)
plt.ylabel('Irreversibility ($x10^{-3}$)', fontsize=10)
#plt.title('Irreversibility by coupling of modelled data,\n including empirical data at optimal GoF')
plt.ylim(all_ymin - 0.2 * all_y_range, all_ymax + 0.5 * all_y_range)  # Increase ylim to lower the bars

# Add a legend
plt.legend()

# Apply the scientific formatter to the y-axis
formatter = FuncFormatter(scientific_single(exponent))
plt.gca().yaxis.set_major_formatter(formatter)

# Add descriptors next to each dot
for i in range(len(disorders_states)):
    plt.text(ssim_tenet_emp[i, 1], ssim_tenet_emp[i, 0], disorders_states[i], ha='right', va='bottom', fontsize = 10)
    
# Add a legend
plt.legend(loc='best', fontsize=8)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
    

# Save the plot in high resolution
plt.savefig('irreversibility_model_emp_DOC.png', dpi=300)

# Display the plot
plt.show()
        
        
#%% Analysis of structural dissimilarity Control dataset

# Create the x-axis values based on the coupling parameter range
coupling_parameter = np.arange(step,gmax+step,step)  # Range from 0.2 to 4 with 0.02 increments

# Compute mean and standard deviation
mean_values = np.mean(fc_fit_ssim_array[wakefulness_indices], axis=1)
std_values = np.std(fc_fit_ssim_array[wakefulness_indices], axis=1)

# Plot each line with standard deviation band
for i in range(len(wakefulness_states)):
    color = plt.cm.tab10(i)  # Get color from the default tab10 colormap
    plt.plot(coupling_parameter, fc_fit_ssim_array[i], label=wakefulness_states[i], color=color)
    #plt.fill_between(coupling_parameter, mean_values[i] - std_values[i], mean_values[i] + std_values[i],
    #                alpha=0.2, color=color)

# Plot the lowest structural dissimilarity values as dotted lines
for i in range(len(wakefulness_states)):
    color = plt.cm.tab10(i)  # Get color from the default tab10 colormap
    plt.axvline(x=coupling_parameter[wakefulness_min_index[i]], linestyle='dotted', color=color, alpha=0.7)

# Add labels and title
plt.xlabel('Coupling parameter')
plt.ylabel('Structural Dissimilarity')
#plt.title('Goodness of Fit')

# Add a legend
plt.legend()

# Save the plot in high resolution
plt.savefig('structural_dissimilarity.png', dpi=300)

# Add a legend
plt.legend(loc='best', fontsize=10)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
    

# Display the plot
plt.show()

print(mean_values)

#%% Analysis of structural dissimilarity DOC dataset

# Create the x-axis values based on the coupling parameter range
coupling_parameter = np.arange(step,gmax+step,step)  # Range from 0.2 to 4 with 0.02 increments

# Compute mean and standard deviation
mean_values = np.mean(fc_fit_ssim_array[disorders_indices], axis=1)
std_values = np.std(fc_fit_ssim_array[disorders_indices], axis=1)

# Plot each line with standard deviation band
for i in range(len(disorders_states)):
    color = plt.cm.tab10(i)  # Get color from the default tab10 colormap
    plt.plot(coupling_parameter, fc_fit_ssim_array[i], label=disorders_states[i], color=color)
    #plt.fill_between(coupling_parameter, mean_values[i] - std_values[i], mean_values[i] + std_values[i],
     #                alpha=0.2, color=color)

# Plot the lowest structural dissimilarity values as dotted lines
for i in range(len(disorders_states)):
    color = plt.cm.tab10(i)  # Get color from the default tab10 colormap
    plt.axvline(x=coupling_parameter[disorders_min_index[i]], linestyle='dotted', color=color, alpha=0.7)

# Add labels and title
plt.xlabel('Coupling parameter')
plt.ylabel('Structural Dissimilarity')
#plt.title('Goodness of Fit')

# Add a legend
plt.legend()

# Save the plot in high resolution
plt.savefig('structural_dissimilarity_disorders.png', dpi=300)

# Add a legend
plt.legend(loc='best', fontsize=10)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
    

# Display the plot
plt.show()

print(std_values)
#%% Are the modelled states at the optimal coupling from different distributions?

# Create the new array by subsetting tenet_model_array using min_index
subset_array = tenet_model_array[0][wakefulness_min_index.flatten(), :]

# Compute the number of consciousness states
num_states = len(wakefulness_states)

# Print the header
print("Wilcoxon Ranksums results:")

# Perform the Wilcoxon rank-sum test for each pair of consciousness states
for i in range(num_states):
    for j in range(i + 1, num_states):
        state1 = wakefulness_states[i]
        state2 = wakefulness_states[j]
        
        # Perform the Wilcoxon rank-sum test on each pair
        stat, p = ranksums(subset_array[i], subset_array[j])
        
        # Print the results
        print(f"{state1} and {state2}: stat = {stat:.5f}, p = {p:.5f}")

#%% Are the modelled states at the optimal coupling from different distributions?

# Create the new array by subsetting tenet_model_array using min_index
subset_array = tenet_model_array[0][disorders_min_index.flatten(), :]

# Compute the number of consciousness states
num_states = len(disorders_states)

# Print the header
print("Wilcoxon Ranksums results:")

# Perform the Wilcoxon rank-sum test for each pair of consciousness states
for i in range(num_states):
    for j in range(i + 1, num_states):
        state1 = disorders_states[i]
        state2 = disorders_states[j]
        
        # Perform the Wilcoxon rank-sum test on each pair
        stat, p = ranksums(subset_array[i], subset_array[j])
        
        # Print the results
        print(f"{state1} and {state2}: stat = {stat:.5f}, p = {p:.5f}")


#%% Functional connectivity plots empirical

# Load the data
data = np.load("fc_emp_model.npz", allow_pickle=True)
fc_emp_mean_array = data["fc_emp_mean_array"]

# Define the consciousness state labels
consciousness_states = ['CNT', 'MCS', 'N3', 'UWS', 'W']

# Define indices
wakefulness_indices = [4, 2]  # Indices for 'W' and 'N3' in the consciousness_states list
disorders_indices = [0, 1, 3]  # Indices for 'CNT', 'MCS', and 'UWS' in the consciousness_states list

# Separate the datasets based on consciousness states
wakefulness_states = ['W', 'N3']
disorders_states = ['CNT', 'MCS', 'UWS']

# Extract data for the specified indices
wakefulness_fc_array = fc_emp_mean_array[wakefulness_indices]
disorders_fc_array = fc_emp_mean_array[disorders_indices]

# Create a figure for the heatmaps
fig, axs = plt.subplots(2, max(len(wakefulness_fc_array), len(disorders_fc_array)), figsize=(12, 12))

# Plot heatmaps for wakefulness states
fig, axs = plt.subplots(1, len(wakefulness_fc_array), figsize=(12, 6))
for i, fc in enumerate(wakefulness_fc_array):
    im = axs[i].imshow(fc, cmap='RdBu_r', interpolation='nearest', vmin=0, vmax=1)
    axs[i].set_title(wakefulness_states[i])

# Hide x labels and tick labels for top plots and y ticks for right plots.
for axis in axs.flat:
    axis.label_outer()

fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
#plt.suptitle('Mean empirical functional connectivity of Wakefulness and Sleep')

# Save the plot in high resolution
plt.savefig('fc_heatmap_emp_control.png', dpi=300)

plt.show()

# Plot heatmaps for disorders states
fig, axs = plt.subplots(1, len(disorders_fc_array), figsize=(12, 6))
for i, fc in enumerate(disorders_fc_array):
    im = axs[i].imshow(fc, cmap='RdBu_r', interpolation='nearest', vmin=0, vmax=1)
    axs[i].set_title(disorders_states[i])

# Hide x labels and tick labels for top plots and y ticks for right plots.
for axis in axs.flat:
    axis.label_outer()

fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
#plt.suptitle('Mean empirical functional connectivity of Disorders of Consciousness')

# Save the plot in high resolution
plt.savefig('fc_heatmap_emp_DOC.png', dpi=300)

plt.show()

#%% Structural Connectivity Plot

data = np.load("structural_connectivity.npz", allow_pickle=True)
struct_connect = data["struct_connect"]

# Create a figure for the heatmap
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the heatmap for structural connectivity
im = ax.imshow(struct_connect, cmap='RdBu_r', interpolation='nearest', vmin=0, vmax=1)

# Set title
#ax.set_title('Structural Connectivity')

# Hide x labels and tick labels for top plots and y ticks for right plots.
ax.label_outer()

fig.colorbar(im, ax=ax, shrink=0.95)

# Save the plot in high resolution
plt.savefig('struct_connect.png', dpi=300)

plt.show()

#%% Functional connectivity simulated

# Load the data
data = np.load("fc_emp_model.npz", allow_pickle=True)
fc_simul_mean = data["fc_simul_mean"]

# Extract the 90x90 matrices for each consciousness state
wakefulness_fc_simul_array = fc_simul_mean[wakefulness_min_index]
disorders_fc_simul_array = fc_simul_mean[disorders_min_index]

# Plot heatmaps for wakefulness states
fig, axs = plt.subplots(1, len(wakefulness_fc_simul_array), figsize=(12, 6))
for i, fc in enumerate(wakefulness_fc_simul_array):
    im = axs[i].imshow(fc, cmap='RdBu_r', interpolation='nearest', vmin=0, vmax=1)
    axs[i].set_title(wakefulness_states[i])

# Hide x labels and tick labels for top plots and y ticks for right plots.
for axis in axs.flat:
    axis.label_outer()

fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
#plt.suptitle('Mean functional connectivity of Control and Wakefulness at optimal GoF (Simulated)')

# Save the plot in high resolution
plt.savefig('fc_heatmap_simul_control.png', dpi=300)

plt.show()

# Plot heatmaps for disorders states
fig, axs = plt.subplots(1, len(disorders_fc_simul_array), figsize=(12, 6))
for i, fc in enumerate(disorders_fc_simul_array):
    im = axs[i].imshow(fc, cmap='RdBu_r', interpolation='nearest', vmin=0, vmax=1)
    axs[i].set_title(disorders_states[i])

# Hide x labels and tick labels for top plots and y ticks for right plots.
for axis in axs.flat:
    axis.label_outer()

fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
#plt.suptitle('Mean functional connectivity of Disorders of Consciousness at optimal GoF (Simulated)')

# Save the plot in high resolution
plt.savefig('fc_heatmap_simul_doc.png', dpi=300)

plt.show()
