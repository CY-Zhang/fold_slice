#!/bin/bash
#SBATCH --job-name=test_job            # Job name
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cz489@cornell.edu  # Where to send mail
#SBATCH --nodes=1                      # number of nodes requested
#SBATCH --ntasks=1                     # number of tasks to run in parallel
#SBATCH --cpus-per-task=32              # number of CPUs required for each task
#SBATCH --gres=gpu:2g.20gb:1                   # number of GPUs required
#SBATCH --time=10:00:00                # Time limit hrs:min:sec
#SBATCH --output=job_%j.log            # Standard output and error log

pwd; hostname; date
echo "SLURM_ARRAY JOB ID is $SLURM_ARRAY_JOB_ID."
echo "SLURM_ARRAY TASK ID is $SLURM_ARRAY_TASK_ID"

module load matlab/R2021a
module load cuda/11.5

EXEPATH='/home/fs01/cz489/fold_slice/ptycho/'
PARFILE='/home/fs01/cz489/ptychography/jobs/mixed_state_example/parameter_mixed_state.txt'

matlab -nodisplay -nosplash -r "cd ~;\
	cd $EXEPATH;\
	prepare_data('$PARFILE');\
	run_mixed_states('$PARFILE');\
	exit"

date
