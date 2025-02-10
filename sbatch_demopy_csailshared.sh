#!/bin/bash
#SBATCH --job-name=demopy_csailshared
#SBATCH --partition=csail-shared
#SBATCH --qos=lab-free-cycles
#SBATCH --exclude=tart,strudel,pizzelle,oliva-titanrtx-1,oliva-titanrtx-3,lebkuchen,gpu20-4.drl

#SBATCH --qos=lab-free

#SBATCH --exclude=oliva-titanrtx-1,oliva-titanrtx-3,gpu20-4.drl

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --error=slurm_error-%A_%a.err
#SBATCH --requeue

# if you don't give input arguments, this script will fail

# activate mamba scripts
echo $HOME
export CONDARC=/data/pulkitag/models/rli14/.condarc

source /data/pulkitag/models/rli14/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate -p "/data/pulkitag/hamer_diffusion_policy_project/venvs/hawor_nfs_mambaenv"

export PYTHONPATH=/data/pulkitag/hamer_diffusion_policy_project/projects/HaWoR:$PYTHONPATH

cd /data/pulkitag/hamer_diffusion_policy_project/projects/HaWoR

#for start_idx in `(0 2 4 6)`; do

# use first cmd line argument
start_idx=$1
mp_total=$2

mp_idx=$((start_idx))

srun --ntasks=1 bash -c "python3 demo_distributed_processing.py --image_root=/data/pulkitag/models/rli14/data/egoexo_hand_gen/image_portrait_view/undistorted/09102024_egoexoval_full --label_root=/data/pulkitag/hamer_diffusion_policy_project/labels/02082025_egoexo_val_hands23 --hands_bbox_root_dir=/data/pulkitag/hamer_diffusion_policy_project/labels/10312024_egoexoval_full_hands23_bbox --allowed_parent_tasks 'Bike Repair' 'Cooking' --mp_idx=$mp_idx --mp_total=$mp_total"
#srun --ntasks=1 bash -c "python3 demo_distributed_processing.py --image_root=/data/pulkitag/models/rli14/data/egoexo_hand_gen/image_portrait_view/undistorted/09102024_egoexotrain_full --label_root=/data/pulkitag/hamer_diffusion_policy_project/labels/02082025_egoexo_train_hands23 --hands_bbox_root_dir=/data/pulkitag/hamer_diffusion_policy_project/labels/10312024_egoexotrain_full_hands23_bbox --allowed_parent_tasks 'Bike Repair' 'Cooking' --mp_idx=$mp_idx --mp_total=$mp_total"

wait