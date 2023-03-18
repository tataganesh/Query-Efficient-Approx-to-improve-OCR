from eval_prep import EvalPrep
from argparse import Namespace

def prep_eval(prep_path, dataset, data_path, ocr):
    """Script to evaluate prep model
    """
    
    eval_info = {
        "show_txt": False, 
        "show_img": False,
        "prep_path": prep_path,
        "dataset": dataset,
        "batch_size": 64,
        "data_base_path": data_path,
        "ocr": ocr,
        "show_orig": False
    }
    
    ns = Namespace(**eval_info)
    evaluator = EvalPrep(ns)
    accuracy, cer = evaluator.eval()
    return {"test_accuracy": accuracy, "test_cer": cer}