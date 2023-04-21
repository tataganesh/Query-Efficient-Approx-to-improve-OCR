#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=11:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_area_logs/%j.out
#SBATCH --array=1

# EXP_NUM=1
EXP_NUM=$((460+${SLURM_ARRAY_TASK_ID}))
echo "Running Experiment $EXP_ID"

module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate

#wandb disabled
#wandb offline
wandb login $WANDB_API_KEY

# OCR="Tesseract"
OCR="EasyOCR"

cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
DATA_PATH="$SLURM_TMPDIR/data"
if [ ! -d $DATA_PATH ]
then
    echo "VGG Dataset extraction started"
    cp /home/ganesh/projects/def-nilanjan/ganesh/datasets/vgg.zip $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip vgg.zip >> /dev/null
    mv vgg data
    echo "VGG Dataset extracted"
else
    echo "VGG Dataset exists"
fi
cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
BATCH_SIZE=64
EPOCH=50
EXP_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/"
CRNN_MODEL_PATH="/home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_8/crnn_warmup/crnn_model_49"
CKPT_BASE_PATH="/home/ganesh/scratch/experiment_$EXP_NUM/ckpts"
CER_JSON_PATH="/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/cer_data_utils/vgg_cers.json"
mkdir -p $CKPT_BASE_PATH

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" job_array_vgg)
eval "$command"
