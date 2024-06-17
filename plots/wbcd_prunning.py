import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/scratch/Behrad/repos/BAOELM/BAoELM2_prev/BAoELM/results/results_pruning_wbcd_embeded_elm.csv'
data = pd.read_csv(file_path)

# Extracting relevant data and converting columns to appropriate data types
data['HIDDEN_LYR_SIZE'] = data['HIDDEN_LYR_SIZE'].astype(int)
data['POISON_PERCENTAGE'] = data['POISON_PERCENTAGE'].astype(float)
data['PRUNE_RATE'] = data['PRUNE_RATE'].astype(float)

data = data[data['POISON_PERCENTAGE'] == 5.0]

# Unique poison percentages for separate plots
prune_rates = data['PRUNE_RATE'].unique()

# Creating the plot
fig, axs = plt.subplots(len(prune_rates), 1, figsize=(5, 15), sharex=True)

# Plotting data for each prune rate
for i, prune_rate in enumerate(prune_rates):
    subset = data[data['PRUNE_RATE'] == prune_rate]
    axs[i].plot(subset['HIDDEN_LYR_SIZE'], subset['TEST_ACCURACY'], label='Test Accuracy', marker='o')
    axs[i].plot(subset['HIDDEN_LYR_SIZE'], subset['BD_TEST_ACCURACY'], label='BD Test Accuracy', marker='x')
    axs[i].set_title(f'Prune Rate: {prune_rate}')
    axs[i].set_xlabel('Hidden Layer Size')
    axs[i].set_ylabel('Accuracy')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig('mp_prunning_wbcd.pdf')

