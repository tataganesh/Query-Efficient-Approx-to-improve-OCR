#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=00:30:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_patch_logs/eval/%j.out


module load StdEnv/2020 tesseract/4.1.0
VENV_PATH="/home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate"
source $VENV_PATH
PROJECT_BASE_PATH=" /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR"

# Command-line arguments
DATASET_NAME=$1
OCR=$2

echo "Dataset=$1 OCR=$2 results"

DATA_PATH="$SLURM_TMPDIR/$DATASET_NAME"
rm -rf "$DATA_PATH"

if [ ! -d "$DATA_PATH" ]
then
    echo "$DATASET_NAME Dataset extraction started"
    cp "/home/ganesh/projects/def-nilanjan/ganesh/datasets/$DATASET_NAME.zip" $SLURM_TMPDIR/
    cd $SLURM_TMPDIR || { echo "Slurm Tmpdir does not exist. Exiting.."; exit 1; }
    unzip $DATASET_NAME.zip >> /dev/null
    echo "$DATASET_NAME Dataset extracted"
else
    echo "$DATASET_NAME Dataset exists"
fi

cd $PROJECT_BASE_PATH || { echo "$PROJECT_BASE_PATH does not exist. Exiting.."; exit 1; }

# shellcheck disable=SC2317 
function run_exp() {

local exp_id="$1"
echo "Experiment $exp_id"
shift
local exps=("$@")
for i in "${exps[@]}";
do
    echo "Preprocessor $i"
    python -u eval_prep.py --prep_path "/home/ganesh/scratch/experiment_$exp_id/ckpts/Prep_model_$i" --dataset $DATASET_NAME --data_base_path $DATA_PATH --ocr $OCR
done

}



# exps=(45_67.99)
# run_exp 500 "${exps[@]}" 

# exps=(45_67.48)
# run_exp 501 "${exps[@]}" # both

exps=(45_67.50)
run_exp 502 "${exps[@]}" # 49

# exps=(40_64.46)
# run_exp 503 "${exps[@]}" #  48

# exps=(37_58.17)
# run_exp 478 "${exps[@]}" #  41

# exps=(33_56.79)
# run_exp 479 "${exps[@]}" #  33


# exit 0
# EOT