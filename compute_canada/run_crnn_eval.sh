#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=1:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/crnn_warmup_logs/eval/%j.out


module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate



date +%c
DATA_PATH="$SLURM_TMPDIR/data"
DATASET_NAME="gcp_textareas"
if [ ! -d $DATA_PATH ]
then
    echo "$DATASET_NAME extraction started"
    # cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/gcloud_vision/exp249_prep/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd $SLURM_TMPDIR
    unzip $DATASET_NAME.zip >> /dev/null
    mv $DATASET_NAME data
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi
# cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
for i in 49
do
    echo "Evaluating CRNN ckpt $i"
    python -u -Wignore eval_crnn.py --crnn_path /home/ganesh/projects/def-nilanjan/ganesh/experiment_artifacts/experiment_431/crnn_warmup/ --dataset pos_textarea --crnn_model_name "crnn_model_$i" --data_base_path $SLURM_TMPDIR
    echo "\n\n"
done
date +%c

