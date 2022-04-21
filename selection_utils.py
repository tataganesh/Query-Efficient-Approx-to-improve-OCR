from abc import ABCMeta, abstractmethod
import torch
import json
from copy import deepcopy
import traceback
class DataSampler(metaclass=ABCMeta):
    def __init__(self, discount_factor=0):
        self.discount_factor = discount_factor  

    @abstractmethod
    def query(self):
        pass


class RandomSampler(DataSampler):
    def query(self, images, labels, num_samples, names=None):
        """Get 

        Args:
            images (torch.tensor): Input images
            labels (torch.tensor): Input labels
            subset (int): Number of random samples.

        Returns:
            tuple: Return subset of images and labels. Chosen randomly. 
        """    
        num_images = images.shape[0]
        rand_indices = torch.randperm(num_images)[:num_samples]
        return images[rand_indices], [labels[i] for i in rand_indices], rand_indices




class UniformCerSampler(DataSampler):
    def __init__(self, cers):
        self.cers = cers
        
    def query(self, images, labels, num_samples, names):
        """Get 

        Args:
            images (torch.tensor): Input images
            labels (torch.tensor): Input labels
            subset (int): Number of random samples.

        Returns:
            tuple: Return subset of images and labels. Chosen randomly. 
        """    
        num_images = images.shape[0]
        half_num_images = int(num_images/2)
        half_num_samples = int(num_samples/2)
        image_cers = list()
        for name in names:
            if name not in self.cers:
                image_cers.append(0.0)
            else:
                image_cers.append(self.cers[name])
        image_cers = torch.tensor(image_cers)
        sorted_indices = torch.argsort(image_cers, descending=True)
        if num_samples == 1:
            selection_idx = torch.tensor([sorted_indices[half_num_images]])
        else:
            before_half = half_num_images- half_num_samples
            after_half = before_half + num_samples
            selection_idx = sorted_indices[before_half: after_half]
        return images[selection_idx], [labels[i] for i in selection_idx], selection_idx
            

    def update_cer(self, cers, names):
        for i in range(len(cers)):
            self.cers[names[i]] = cers[i]




def datasampler_factory(sampling_method):
    method_mapping = {'uniformCER': UniformCerSampler, 'random': RandomSampler}
    return method_mapping[sampling_method]


if __name__ == "__main__":
    cls_sampler = datasampler_factory("uniformCER")
    sampler = cls_sampler('/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/all_cers_textarea.json')
    names = ['0_6.600_receipt_00206.png', '1_KEMBALI_receipt_00206.png','2_70.000_receipt_00206.png','3_TUNAI_receipt_00206.png','4_63,400_receipt_00206.png']
    sampler.query(torch.tensor([1,2,3, 4,5]), torch.tensor([1,2,3, 4,5]), 4, names)

    cls_sampler = datasampler_factory("random")
    sampler = cls_sampler()
    sampler.query(torch.tensor([1,2,3, 4,5]), torch.tensor([1,2,3, 4,5]), 4)
