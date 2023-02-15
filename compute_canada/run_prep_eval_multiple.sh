#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8000       # Memory proportional to GPUs: 32000 Cedar, 47000 B�luga, 64000 Graham.
#SBATCH --time=0:20:00     # DD-HH:MM:SS
#SBATCH --output=/home/ganesh/projects/def-nilanjan/ganesh/nn_area_logs/eval/%j.out


module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate


echo "EasyOCR VGG results"

EXP_ID=259
DATA_PATH="$SLURM_TMPDIR/data"

rm -rf $DATA_PATH
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
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
# [ ! -f "/home/ganesh/scratch/experiment_$EXP_ID/ckpts/Prep_model_4" ] && echo "File not found!" 
# for i in 38 45 48 49
# do
#     echo "Running $i preprocessor"
#     # python -u eval_prep.py --prep_path "/home/ganesh/scratch/experiment_$EXP_ID/ckpts/" --dataset pos --prep_model_name "Prep_model_$i" --data_base_path $SLURM_TMPDIR
#     python -u eval_prep.py --prep_path "/home/ganesh/scratch/experiment_$EXP_ID/ckpts/" --dataset vgg --prep_model_name "Prep_model_$i" --data_base_path $SLURM_TMPDIR
# done

function run_exp() {
local exp_id="$1"
echo "Experiment $exp_id"
shift
local exps=("$@")
for i in "${exps[@]}";
do
    echo "Preprocessor $i"
    python -u eval_prep.py --prep_path "/home/ganesh/scratch/experiment_$exp_id/ckpts/" --dataset pos --prep_model_name "Prep_model_$i" --data_base_path $SLURM_TMPDIR --ocr EasyOCR
done

}



exps=(48)
run_exp 360 "${exps[@]}" # 46

exps=(49)
run_exp 361 "${exps[@]}" # both

# exps=(47)
# run_exp 334 "${exps[@]}" # 49

# exps=(49)
# run_exp 335 "${exps[@]}" #  48

# exps=(41 44)
# run_exp 336 "${exps[@]}" #  42

# exps=(47)
# run_exp 337 "${exps[@]}" #  45

# exps=(45)
# run_exp 338 "${exps[@]}" #  47

# exps=(43)
# run_exp 339 "${exps[@]}" #  48

# exps=(49)
# run_exp 340 "${exps[@]}" #  41

# exps=(33 34)
# run_exp 325 "${exps[@]}" #  33


# exps=(45 49)
# run_exp 304 "${exps[@]}" #  POS 4% + random - 45

# exps=(43 45)
# run_exp 305 "${exps[@]}" #  POS 4% +  random + tracking - 45


# VGG

# exps=(41)
# run_exp 350 "${exps[@]}" # Highest - 27, later - 47

# exps=(45)
# run_exp 342 "${exps[@]}" # 48

# exps=(48)
# run_exp 343 "${exps[@]}" # 45

# exps=(46)
# run_exp 344 "${exps[@]}" #  48

# exps=(45)
# run_exp 345 "${exps[@]}" #  46

# exps=(27 31)
# run_exp 346 "${exps[@]}" #  27 highest

# exps=(36)
# run_exp 347 "${exps[@]}" #  49

# exps=(43)
# run_exp 348 "${exps[@]}" #  41

# exps=(40)
# run_exp 349 "${exps[@]}" #  POS 2.5% + cer