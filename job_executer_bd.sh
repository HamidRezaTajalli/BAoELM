#! /bin/bash
#MIT License
#
#Copyright (c) 2023 Abraham J. Basurto Becerra
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#SBATCH --job-name=BAoELM
#SBATCH --account=icis
#SBATCH --partition=icis
#SBATCH --qos=icis-large                         # see https://wiki.icis-intra.cs.ru.nl/Cluster#Job_Class_Specifications
#SBATCH --nodes=1                                # node count
##SBATCH --nodelist=cn114                        # run in this specific node
#SBATCH --ntasks=1                               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2                        # cpu-cores per task
##SBATCH --mem-per-cpu=16G                         # memory per cpu-core
#SBATCH --mem=8G                                # memory per node
##SBATCH --gres=gpu:rtx_a5000:1                   # assign 1 RTX A5000 GPU card
#SBATCH --time=0-23:00:00
##SBATCH --output=/home/%u/log/slurm/%j.out       # stdout output file
##SBATCH --error=/home/%u/log/slurm/%j.err        # stderr output file
#SBATCH --mail-type=END,FAIL                     # send email when job ends or fails
#SBATCH --mail-user=hamidreza.tajalli@ru.nl      # email address




# Activate Conda environment
source /home/htajalli/.bashrc
conda activate AISY

python main_direct_bd.py --dataname mnist --elmtype poelm
#python main_direct_bd.py --dataname mnist --elmtype drop-elm
#python main_direct_bd.py --dataname mnist --elmtype telm
#python main_direct_bd.py --dataname mnist --elmtype mlelm
#
#python main_direct_bd.py --dataname fmnist --elmtype poelm
#python main_direct_bd.py --dataname fmnist --elmtype drop-elm
#python main_direct_bd.py --dataname fmnist --elmtype telm
#python main_direct_bd.py --dataname fmnist --elmtype mlelm
#
#python main_direct_bd.py --dataname svhn --elmtype poelm
#python main_direct_bd.py --dataname svhn --elmtype drop-elm
#python main_direct_bd.py --dataname svhn --elmtype telm
#python main_direct_bd.py --dataname svhn --elmtype mlelm
