from abc import ABCMeta, abstractmethod
from re import L
import torch
import json
from copy import deepcopy
import traceback
import numpy as np
import random

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
    def __init__(self, cers, discount_factor=1):
        self.cers = cers
        self.discount_factor = discount_factor
        
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
        half_num_images = int(num_images/2) + num_images % 2
        image_cers = list()
        for name in names:
            if name in self.cers:
                image_cers.append(self.cers[name])
        image_cers = torch.tensor(image_cers)
        sorted_indices = torch.argsort(image_cers, descending=True)
        alternating_indices = torch.zeros_like(sorted_indices)
        even_indices = np.arange(0, sorted_indices.shape[0], 2)
        odd_indices = np.arange(1, sorted_indices.shape[0], 2)
        alternating_indices[even_indices] = sorted_indices[:half_num_images]
        second_half_indices = sorted_indices[half_num_images:]
        inv_idx = torch.arange(second_half_indices.shape[0]-1, -1, -1).long()
        alternating_indices[odd_indices] = second_half_indices[inv_idx]
        selection_idx = alternating_indices[:num_samples]
        return images[selection_idx], [labels[i] for i in selection_idx], selection_idx
            

    def update_cer(self, cers, names):
        for i in range(len(cers)):
            if names[i] not in self.cers:
                continue
            self.cers[names[i]] = self.discount_factor * cers[i] +  (1 - self.discount_factor) * self.cers[names[i]]


class CerRangeSampler(DataSampler):
    def __init__(self, cers, discount_factor=1):
        self.cers = cers
        self.discount_factor = discount_factor
        
    def query(self, images, labels, num_samples, names):
        """Get 

        Args:
            images (torch.tensor): Input images
            labels (torch.tensor): Input labels
            subset (int): Number of random samples.

        Returns:
            tuple: Return subset of images and labels. Chosen randomly. 
        """    
        image_cers = list()
        selection_idx = torch.tensor([], dtype=torch.long)
        for name in names:
            if name in self.cers:
                image_cers.append(self.cers[name])
        image_cers = torch.tensor(image_cers)
        if image_cers.shape[0] != 0:
            cer_random_points = (image_cers.max() - image_cers.min()) * torch.rand(num_samples) + image_cers.min()
            # cer_diff = torch.abs(image_cers.unsqueeze(1) - cer_random_points.unsqueeze(0))
            selection_idx = torch.zeros(num_samples, dtype=torch.long)
            image_cers_copy = torch.clone(image_cers)
            for i, point in enumerate(cer_random_points):
                index = torch.argmin(torch.abs(point - image_cers_copy))
                selection_idx[i] = index
                image_cers_copy[i] = 100
        return images[selection_idx], [labels[i] for i in selection_idx], selection_idx
            

    def update_cer(self, cers, names):
        for i in range(len(cers)):
            if names[i] not in self.cers:
                continue
            self.cers[names[i]] = self.discount_factor * cers[i] +  (1 - self.discount_factor) * self.cers[names[i]]



class UniformSamplerGlobal(DataSampler):
    def __init__(self, cers, num_samples):
        self.cers = cers
        self.num_samples = num_samples
        self.selected_indices = np.zeros(num_samples, dtype=np.int32)
        self.selected_samplenames = dict()

    def select_samples(self):
        self.selected_samplenames.clear()
        cer_keys = list(self.cers.keys())
        cer_values = np.array(list(self.cers.values()))
        sorted_cer_indices = np.argsort(cer_values)
        for i, split in enumerate(np.array_split(sorted_cer_indices, self.num_samples)):
            self.selected_indices[i] = np.random.choice(split)
            selected_samplename = cer_keys[self.selected_indices[i]]
            self.selected_samplenames[selected_samplename] = True 


    def query(self, images, labels, num_samples=-1, names=None):
        selection_idx = list()
        for i, name in enumerate(names):
            if name in self.selected_samplenames:
                selection_idx.append(i)
        selection_idx = torch.tensor(selection_idx).long()
        return images[selection_idx], [labels[i] for i in selection_idx], selection_idx

    def update_cer(self, sample_cers, names):
        for i in range(len(sample_cers)): 
            if names[i] not in self.cers:
                #print(f"{names[i]} not found in cer json")
                continue
            self.cers[names[i]] = sample_cers[i]


class RandomSamplerGlobal(DataSampler):
    def __init__(self, cers, num_samples):
        self.cers = cers
        self.num_samples = num_samples
        self.selected_samplenames = dict()

    def select_samples(self):
        self.selected_samplenames.clear()
        cer_keys = list(self.cers.keys())
        samplenames = random.sample(cer_keys, self.num_samples)
        for name in samplenames:
            self.selected_samplenames[name] = True 


    def query(self, images, labels, num_samples=-1, names=None):
        selection_idx = list()
        for i, name in enumerate(names):
            if name in self.selected_samplenames:
                selection_idx.append(i)
        selection_idx = torch.tensor(selection_idx).long()
        return images[selection_idx], [labels[i] for i in selection_idx], selection_idx

    def update_cer(self, sample_cers, names):
        for i in range(len(sample_cers)): 
            if names[i] not in self.cers:
                #print(f"{names[i]} not found in cer json")
                continue
            self.cers[names[i]] = sample_cers[i]

def datasampler_factory(sampling_method):
    method_mapping = {'uniformCER': UniformCerSampler, 'random': RandomSampler, "uniformCERglobal": UniformSamplerGlobal, "randomglobal": RandomSamplerGlobal, "rangeCER": CerRangeSampler}
    return method_mapping[sampling_method]


if __name__ == "__main__":
    cls_sampler = datasampler_factory("uniformCER")
    sampler = cls_sampler('/home/ganesh/projects/def-nilanjan/ganesh/Gradient-Approx-to-improve-OCR/all_cers_textarea.json')
    names = ['0_6.600_receipt_00206.png', '1_KEMBALI_receipt_00206.png','2_70.000_receipt_00206.png','3_TUNAI_receipt_00206.png','4_63,400_receipt_00206.png']
    sampler.query(torch.tensor([1,2,3, 4,5]), torch.tensor([1,2,3, 4,5]), 4, names)

    cls_sampler = datasampler_factory("random")
    sampler = cls_sampler()
    sampler.query(torch.tensor([1,2,3, 4,5]), torch.tensor([1,2,3, 4,5]), 4)
