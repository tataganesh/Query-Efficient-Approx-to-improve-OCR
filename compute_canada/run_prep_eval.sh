#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=1:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/eval/%j.out


module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate



EXP_ID=374

DATA_PATH="$SLURM_TMPDIR/data"
DATASET_NAME="patch_dataset"
# DATASET_NAME="vgg"
if [ ! -d $DATA_PATH ]
then
    echo "$DATASET_NAME Dataset extraction started"
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip $DATASET_NAME.zip >> /dev/null
    mv $DATASET_NAME data
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi
# cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
# python -u eval_prep.py --prep_path ~/scratch/experiment_9/ckpts/ --dataset vgg --prep_model_name Prep_model_26 --data_base_path $SLURM_TMPDIR
# cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
# [ ! -f "/home/ganesh/scratch/experiment_$EXP_ID/ckpts/Prep_model_4" ] && echo "File not found!" 
for i in 47
do
    echo "Running $i preprocessor"
    # python -u eval_prep.py --prep_path "/home/ganesh/scratch/experiment_$EXP_ID/ckpts/" --dataset pos  --prep_model_name "Prep_model_$i" --data_base_path $SLURM_TMPDIR
    python -u eval_prep.py --prep_path "/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_249/ckpts/Prep_model_44" --dataset pos --data_base_path $SLURM_TMPDIR --ocr gvision
done
