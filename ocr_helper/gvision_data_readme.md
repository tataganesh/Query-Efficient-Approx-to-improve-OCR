* Run `run_gvision_label.sh` to preprocess images using a pre-trained preprocessor and optionally call the Google Vision API to get text labels. Refer to `get_labels_gvision.py` for more details. 

* Zip the `train`, `dev` and `test` folders. E.g. `zip -r patch_dataset_train.zip patch_dataset_train`.

* `scp` the files to the location machines for extracting `text strips`. This is done to avoid large number of files in `cedar` since there is a limit to the number of files on disk. 

* Run `gvision_copy_files.py` to copy the GT json files for `dev` and `test` sets. There should be a simpler alternative to obtain bboxes for the preprocessed images. 

* Run `python3 generate_text_strips.py` for generating  `train`, `dev` and `test` folders containing text strips. Zip them and transfer them to Cedar.

* Obtain accuracy of google vision API with given preprocessor using `run_eval_prep.sh`. This does not require the text areas. 

* Train new CRNN model with the preprocessed text strips. Log the model (somewhere?).

* Evaluate performance on new CRNN model. 

