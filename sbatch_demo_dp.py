# launch demopy jobs enmasse
# make sure to change settings in sbatch_demopy_csailshared.sh

import subprocess
import shlex

# let's use 40 for now
num_total_jobs = 60

# lab shared
# run one job per gpu
for start_idx in range(0, num_total_jobs):
    cmd = f"sbatch sbatch_demopy_csailshared.sh {start_idx} {num_total_jobs}"
    # cmd = f"sbatch sbatch_hands23_demobbox_csailshared.sh {start_idx} {num_total_jobs}"
    cmd_args = shlex.split(cmd)
    process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(cmd)