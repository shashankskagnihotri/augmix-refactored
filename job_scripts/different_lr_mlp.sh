#!/bin/bash
echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python augmix_refactored/script/train_mlp.py -lr 0.001
python augmix_refactored/script/train_mlp.py -lr 0.01
python augmix_refactored/script/train_mlp.py -lr 0.1 
python augmix_refactored/script/train_mlp.py -lr 0.5 
python augmix_refactored/script/train_mlp.py -lr 1.0 
python augmix_refactored/script/train_mlp.py -lr 5.0 
python augmix_refactored/script/train_mlp.py -lr 10.0
python augmix_refactored/script/train_mlp.py -lr 0.0001
python augmix_refactored/script/train_mlp.py -lr 0.005
python augmix_refactored/script/train_mlp.py -lr 0.3

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime