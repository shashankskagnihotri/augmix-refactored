#!/bin/bash
echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python augmix_refactored/script/cifar.py --config config/cifar_jsd_scale_augmix.yaml --save-folder ./snapshots/softmax/jsd_scale_augmix --dataset cifar100

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
