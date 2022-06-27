## Steps to create json with text strip cers and subsequently split the dataset based on the CER threshold.

* Run ocr_inference.py on the text area dataset so that you can obtain json with the following format - 
```
{
    "index_label_foldername": cervalue
}

{
    "0_hunting_receipt466": 0.1
}
```
```
 python ocr_inference_patch_dataset.py  --data_base_path $SLURM_TMPDIR --cers_save_path pos_dataset_cers.json
 ``` 
* Use json from ocr_inference.py and pass it to patch_image_mapping.py (along with the dataset folder) to obtain the json of the following format :

```
The format of  all_cers_textarea.json is as follows

{
    "index_label_foldername": cervalue
}
Using the patch dataset information, we walk through the different dataset folders and 
convert the loaded json to the following format - 

{
	"receipt_00040_0.png": {
		"0_1_receipt_00040.png": 2.0,
		"1_Home_receipt_00040.png": 1.0,
		"2_Made_receipt_00040.png": 1.0
	}
}
```

* Use json from patch_image_mapping.py in cer_analysis_datasplit.py. Refer to its docstring for more info.