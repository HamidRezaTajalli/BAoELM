import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/scratch/Behrad/repos/BAOELM/BAoELM2_prev/BAoELM/results/results_bd_model_poisoning_wbcd.csv'
data = pd.read_csv(file_path)

# Extracting relevant data and converting columns to appropriate data types
data['HIDDEN_LYR_SIZE'] = data['HIDDEN_LYR_SIZE'].astype(int)
data['POISON_PERCENTAGE'] = data['POISON_PERCENTAGE'].astype(float)

# Unique poison percentages for separate plots
poison_percentages = data['POISON_PERCENTAGE'].unique()

# Creating the plot
fig, axs = plt.subplots(len(poison_percentages), 1, figsize=(10, 15), sharex=True)

# Plotting data for each poison percentage
for i, poison_percentage in enumerate(poison_percentages):
    subset = data[data['POISON_PERCENTAGE'] == poison_percentage]
    axs[i].plot(subset['HIDDEN_LYR_SIZE'], subset['TEST_ACCURACY'], label='Test Accuracy', marker='o')
    axs[i].plot(subset['HIDDEN_LYR_SIZE'], subset['BD_TEST_ACCURACY'], label='BD Test Accuracy', marker='x')
    axs[i].set_title(f'Poison Percentage: {poison_percentage}')
    axs[i].set_xlabel('Hidden Layer Size')
    axs[i].set_ylabel('Accuracy')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig('mp_asr_wbcd.pdf')

