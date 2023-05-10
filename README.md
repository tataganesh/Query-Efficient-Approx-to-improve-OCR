# Query-Efficient Black Box Approximation for OCR

The repository contains code for training and evaluating the experiments performed in the submission titled "Document Image Cleaning using Budget-Aware Black-Box Approximation". A large part of the code is derived from [Gradient-Approx-to-improve-OCR](https://github.com/paarandika/Gradient-Approx-to-improve-OCR).

## Setup
Create a python virtual environment and install the required packages using
```bash
pip3 install -r requirements.txt
```

## Datasets
The dataset links are as follows:

* [VGG](https://drive.google.com/file/d/1_g5rdNMbwR4MUZORBLw4PUSHWoRqtE2r/view?usp=sharing)
* [POS](https://drive.google.com/file/d/1h4NI8h1FTYuIswbeUz_ICd_VTNfTlEPZ/view)
* [POS Text Areas](https://drive.google.com/file/d/1TL9Kda5l8rSyAt7NB7QblETmk5wSwhLC/view?usp=share_link)

Train, Val and Test splits should be extracted and placed in a folder called "data". 

## Training
An example command to train a preprocessor using the POS dataset is shown below - 

```python
python -u train_nn_patch.py --epoch $EPOCH --data_base_path $DATA_PATH --crnn_model  $CRNN_MODEL_PATH --exp_base_path $EXP_BASE_PATH  --minibatch_subset TopKCER --minibatch_subset_prop 0.95  --inner_limit 1 --inner_limit_skip --cers_ocr_path $CER_JSON_PATH --ocr $OCR
```
Relevant arguments are explained here

* `data_base_path`: Path to folder with dataset in "data" folder. 
* `crnn_model`: Path to pre-trained CRNN model
* `exp_base_path`: Path for saving model checkpoints
* `minibatch_subset`: Used to specify different selection algorithms. (Random=random, TopKCER=TopKCER, UniformCER=rangeCER)
* `minibatch_subset_prop`: Specify the proportion of samples for each OCR is not queried. Here, 0.95 indicates skipping almost 95-96% of samples, hence the OCR is queried for only 4% of samples. 
* `inner_limit`: Number of times the images are jittered. If inner_limit_skip is specified, label tracking is enabled and images are not jittered at all.
* `cers_ocr_path`: Initialize the sample cers with a json file. E.g. [VGG](vgg_dataset_cers.json), [POS](pos_dataset_cers.json)
* `ocr`: Specify the OCR - Tesseract / EasyOCR

To train a preprocessor with the VGG dataset, use `train_nn_area.py` with the same arguments as `train_nn_patch.py`. 


An example command to train a CRNN model is shown below - 

```python
python -u train_crnn.py --batch_size $BATCH_SIZE --epoch $EPOCH --crnn_model_path $CRNN_MODEL_PATH --dataset vgg --data_base_path $DATA_PATH --ocr EasyOCR
```

## Evaluation

`eval_prep.py` is used for evaluating a trained preprocessor. 
```python
python -u eval_prep.py --prep_path $PREP_PATH --dataset pos --prep_model_name $PREP_MODEL_NAME --data_base_path $DATA_PATH --ocr EasyOCR
```

* `prep_path` specifies folder path containing preprocessor checkpoints. 
* `prep_model_name` specifies name of specific model checkpoint to be evaluated. 
* `dataset` specifies pos/vgg dataset. 


## Trained Models
The directory `pretrained_models` contains trained preprocessors and pretrained CRNN models from some experiments. The `preprocessor` directory contains models with name `n_model` where `n` can be 4, 8 or 100 (indicating the query budget). The models in the `preprocessor` directory were obtained using the POS dataset and Tesseract OCR engine. 

### Pending Items

 - [ ] Trained Models
 - [ ] Add colab link
 - [ ] Hugging Face spaces demo
 - [ ] Remove "Train, Val and Test splits should be extracted and placed in a folder called 'data'".
 - [ ] Budget Calculation Explanation
 - [ ] Some results


