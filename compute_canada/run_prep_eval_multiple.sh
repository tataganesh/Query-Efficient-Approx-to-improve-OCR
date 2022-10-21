#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=01:00:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/eval/%j.out


module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate



EXP_ID=166
DATA_PATH="$SLURM_TMPDIR/data"

rm -rf $DATA_PATH
# if [ ! -d $DATA_PATH ]
# then
#     echo "$DATASET_NAME Dataset extraction started"
#     cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
#     cd $SLURM_TMPDIR
#     unzip $DATASET_NAME.zip >> /dev/null
#     mv $DATASET_NAME data
#     echo "$DATASET_NAME Dataset extracted"
# else
#     echo "$DATASET_NAME Dataset exists"
# fimodule load StdEnv/2020 tesseract/4.1.0
# source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate

DATA_PATH="$SLURM_TMPDIR/data"
#DATASET_NAME="patch_dataset"
DATASET_NAME="funsd"
# DATASET_NAME="pos"
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
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
# [ ! -f "/home/ganesh/scratch/experiment_$EXP_ID/ckpts/Prep_model_4" ] && echo "File not found!" 
# for i in 38 45 48 49
# do
#     echo "Running $i preprocessor"
#     # python -u eval_prep.py --prep_path "/home/ganesh/scratch/experiment_$EXP_ID/ckpts/" --dataset pos --prep_model_name "Prep_model_$i" --data_base_path $SLURM_TMPDIR
#     python -u eval_prep.py --prep_path "/home/ganesh/scratch/experiment_$EXP_ID/ckpts/" --dataset vgg --prep_model_name "Prep_model_$i" --data_base_path $SLURM_TMPDIR
# done

run_exp() {
echo "Experiment $1"
for i in $2 $3 $4
do
    echo "Preprocessor $i"
    python -u eval_prep.py --prep_path "/home/ganesh/scratch/experiment_$1/ckpts/" --dataset funsd --prep_model_name "Prep_model_$i" --data_base_path $SLURM_TMPDIR
done

}

run_exp 254 37 39 47 # 50% rangecer pos
#run_exp 250 41 42 46 # 8% rangecer pos wrmup
#run_exp 251 42 44 49 # 8% rangecer pos + no noise warmup
#run_exp 252 39 47 49 # 8% rangecer pos + tracking

