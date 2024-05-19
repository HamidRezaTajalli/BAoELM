import training_bd
from pathlib import Path
import gc
import os
import shutil
import subprocess

# Path to the job_executer.sh template
job_executer_template_path = "job_executer.sh"

saving_path = '.'
n_of_experiments = 1
elm_type_list = ['poelm', 'drop-elm', 'mlelm']
dataset_list = ['wbcd', 'brats']
elm_type_list = ['poelm']
dataset_list = ['brats']

hdlyr_size_list = [500, 1000, 2000, 5000, 8000]
# hdlyr_size_list = [5000, 8000, 10000]

trigger_type = 'badnet'
epsilon_list = [0.5, 1, 2, 5]
trigger_size_list = [(2, 2), (4, 4), (8, 8)]
target_label = 0


for dataset in dataset_list:
    for elm_type in elm_type_list:
        for hdlyr_size in hdlyr_size_list:
            for epsilon in epsilon_list:
                for trigger_size in trigger_size_list:
                    for exp_num in range(n_of_experiments):
                        # Define the job script filename
                        job_script_filename = f"job_{dataset}_{elm_type}_{hdlyr_size}_{epsilon}_{trigger_size[0]}x{trigger_size[1]}_{exp_num}.sh"
                        job_script_path = Path(saving_path) / job_script_filename

                        # Read the template and append the command to run the experiment
                        with open(job_executer_template_path, 'r') as file:
                            content = file.read()

                        # Command to run the experiment
                        command = f"python training_bd.py --exp_num {exp_num} --saving_path {saving_path} --elm_type {elm_type} --dataset {dataset} --hdlyr_size {hdlyr_size} --trigger_type {trigger_type} --target_label {target_label} --poison_percentage {epsilon} --trigger_size {trigger_size[0]}"
                        
                        # Write the new job script
                        with open(job_script_path, 'w') as file:
                            file.write(content)
                            file.write("\n")  # Ensure there's a newline before adding the command
                            file.write(command)

                        # Make the script executable
                        subprocess.run(['chmod', '+x', str(job_script_path)])
                        # Submit the job script to SLURM
                        subprocess.run(['sbatch', str(job_script_path)])


