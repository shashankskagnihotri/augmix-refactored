#!/bin/bash
echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

for filename in job_scripts/*.sh; do
    sbatch -p gpu --gres=gpu:1 -t 23:59:59 --ntasks=1 --cpus-per-task=16 --mem=50G $filename
    echo $filename
done
## sbatch -p gpu --gres=gpu:1 -t 23:59:59 --ntasks=1 --cpus-per-task=16 --mem=50G job_scripts/

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
