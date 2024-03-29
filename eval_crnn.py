import torch
import argparse
import os

import torchvision.transforms as transforms

from datasets.patch_dataset import PatchDataset
from datasets.img_dataset import ImgDataset
from utils import show_img, compare_labels, get_text_stack, get_ocr_helper, get_char_maps, pred_to_string
from transform_helper import PadWhite
import properties as properties


class EvalCRNN():

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.show_txt = args.show_txt
        self.show_img = args.show_img
        self.crnn_model_name = args.crnn_model_name
        self.crnn_model_path = args.crnn_path
        self.ocr_name = args.ocr
        self.dataset_name = args.dataset
        self.show_orig = args.show_orig

        if self.dataset_name == 'vgg':
            self.test_set = os.path.join(args.data_base_path, properties.vgg_text_dataset_test)
            self.input_size = properties.input_size
        elif self.dataset_name == 'pos':
            self.test_set = os.path.join(args.data_base_path, properties.patch_dataset_test)
            self.input_size = properties.input_size
        elif self.dataset_name == 'pos_textarea':
            self.test_set = os.path.join(args.data_base_path, properties.pos_text_dataset_test)
            self.input_size = properties.input_size

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.crnn_model = torch.load(os.path.join(
            self.crnn_model_path, self.crnn_model_name)).to(self.device)
        print(f"OCR name - {self.ocr_name}")
        self.ocr = get_ocr_helper(self.ocr_name, is_eval=True)
        print(self.ocr)
        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)

        if self.dataset_name == 'pos':
            self.dataset = PatchDataset(self.test_set, pad=True)
        else:
            transform = transforms.Compose([
                PadWhite(self.input_size),
                transforms.ToTensor(),
            ])
            self.dataset = ImgDataset(
                self.test_set, transform=transform, include_name=True)
            self.loader_eval = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, num_workers=properties.num_workers)

    def _call_model(self, images, labels):
        X_var = images.to(self.device)
        scores = self.crnn_model(X_var)
        out_size = torch.tensor(
            [scores.shape[0]] * images.shape[0], dtype=torch.int)
        y_size = torch.tensor([len(l) for l in labels], dtype=torch.int)
        conc_label = ''.join(labels)
        y = [self.char_to_index[c] for c in conc_label]
        # y = [self.char_to_index[c] if c in self.char_to_index else self.char_to_index[" "] for c in conc_label]
        y_var = torch.tensor(y, dtype=torch.int)
        return scores, y_var, out_size, y_size

    def _print_labels(self, labels, pred, ori):
        print()
        print('{:<25}{:<25}{:<25}'.format("GT Label",
                                          "Label for pred", "Label for original"))
        for i in range(len(labels)):
            try:
                print('{:<25}{:<25}{:<25}'.format(labels[i], pred[i], ori[i]))
            except:
                try:
                    print('{:<25}{:<25}{:<25}'.format(
                        labels[i], "*******", ori[i]))
                except:
                    continue

    def eval_area(self):
        print("Eval with ", self.ocr_name)
        self.crnn_model.eval()
        crnn_correct_count = 0
        ori_correct_count = 0
        ori_cer = 0
        crnn_cer = 0
        counter = 0

        for images, labels, names in self.loader_eval:
            X_var = images.to(self.device)
            scores, y, pred_size, y_size = self._call_model(
                        X_var, labels)            
            # ocr_lbl_pred = self.ocr.get_labels(X_var.cpu())
            ocr_lbl_crnn = pred_to_string(scores.cpu(), labels, self.index_to_char)
            if self.show_orig:
                ocr_lbl_ori = self.ocr.get_labels(images.cpu())
                ori_crt_count, o_cer = compare_labels(ocr_lbl_ori, labels)
                ori_correct_count += ori_crt_count
                ori_cer += o_cer

            if self.show_txt:
                self._print_labels(labels, ocr_lbl_crnn, ocr_lbl_ori)

            crnn_crt_count, crn_cer = compare_labels(
                ocr_lbl_crnn, labels)
            crnn_correct_count += crnn_crt_count
            crnn_cer += crn_cer

            # if self.show_img:
            #     show_img(img_preds.detach().cpu(), "Processed images")
            counter += 1
        print()
        print('Correct count from CRNN: {:d}/{:d} ({:.5f})'.format(
            crnn_correct_count, len(self.dataset), crnn_correct_count/len(self.dataset)))
        if self.show_orig:
            print('Correct count from Tesseract: {:d}/{:d} ({:.5f})'.format(
                ori_correct_count, len(self.dataset), ori_correct_count/len(self.dataset)))
            print('Average CER using Tesseract: {:.5f}'.format(
                ori_cer/len(self.dataset)))
        print('Average CER using CRNN: {:.5f}'.format(
            crnn_cer/len(self.dataset)))

    def eval_patch(self):
        print("Eval with ", self.ocr_name)
        ori_lbl_crt_count = 0
        ori_lbl_cer = 0
        lbl_count = 0
        counter = 0
        crnn_correct_count = 0
        crnn_cer = 0

        for image, labels_dict in self.dataset:
            text_crops, labels = get_text_stack(
                image.detach(), labels_dict, self.input_size)
            lbl_count += len(labels)
            if self.show_orig:
                ocr_labels = self.ocr.get_labels(text_crops)

                ori_crt_count, ori_cer = compare_labels(
                    ocr_labels, labels)
                ori_lbl_crt_count += ori_crt_count
                ori_lbl_cer += ori_cer

            scores, y, pred_size, y_size = self._call_model(
                        text_crops.to(self.device), labels)            
            ocr_lbl_crnn = pred_to_string(scores.cpu(), labels, self.index_to_char)

            crnn_crt_count, crn_cer = compare_labels(
                ocr_lbl_crnn, labels)
            crnn_correct_count += crnn_crt_count

            crnn_cer += crn_cer

            crnn_cer = round(crnn_cer/len(labels), 2)

            if self.show_img:
                show_img(image.cpu())
            # if self.show_txt:
            #     self._print_labels(labels, pred_labels, ocr_labels)
            counter += 1
        print()
        print('Correct count from predicted images: {:d}/{:d} ({:.5f})'.format(
            crnn_correct_count, lbl_count, crnn_correct_count/lbl_count))
        if self.show_orig:
            print('Correct count from original images: {:d}/{:d} ({:.5f})'.format(
                ori_lbl_crt_count, lbl_count, ori_lbl_crt_count/lbl_count))
            print('Average CER from original images: ({:.5f})'.format(
                ori_lbl_cer/lbl_count))
        print('Average CER from predicted images: ({:.5f})'.format(
            crnn_cer/lbl_count))


    def eval(self):
        if self.dataset_name == 'pos':
            self.eval_patch()
        else:
            self.eval_area()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains the CRNN model')
    parser.add_argument('--show_txt', action='store_true',
                        help='prints predictions and groud truth')
    parser.add_argument('--show_img', action='store_true',
                        help='shows each batch of images')
    parser.add_argument('--crnn_path', default=properties.crnn_model_path,
                        help="specify non-default prep model location")
    parser.add_argument('--dataset', default='pos',
                        help="performs training with given dataset [pos, vgg]")
    parser.add_argument('--ocr', default="Tesseract",
                        help="performs training lebels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument("--crnn_model_name",
                        default='prep_tesseract_pos', help='Prep model name')
    parser.add_argument("--batch_size", default=64, type=int,  help='Inference batch size')
    parser.add_argument('--data_base_path',
                        help='Base path training, validation and test data', default=".")
    parser.add_argument('--show_orig', help="Show original flow evaluation", action="store_true")
    
    args = parser.parse_args()
    print(args)
    evaluator = EvalCRNN(args)
    evaluator.eval()
