module load StdEnv/2020 tesseract/4.1.0
source /home/ganesh/projects/def-nilanjan/ganesh/ocr_bb_calls/bin/activate


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
# cd /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
cd  /home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR
for i in 20 25 30 35 40 45 49
do
    echo "Evaluating CRNN ckpt $i"
    python -u -Wignore eval_crnn.py --crnn_path /home/ganesh/scratch/experiment_8/crnn_warmup --dataset vgg --crnn_model_name "crnn_model_$i" --data_base_path $SLURM_TMPDIR
    echo "\n\n"
done