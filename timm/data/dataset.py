""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import time
from multiprocessing import Pool, current_process

import torch
import torch.utils.data as data
from PIL import Image
# from dwt import *
from .readers import create_reader
# import torch_dwt as tdwt
import torchvision.transforms as T
import random
from timm.data.aug2 import auxiliary_argument_policy, auxliary_single_policy, fix_level_auxiliary_argument_policy
from timm.data.constants import PACS_PHOTO_DWT_MEAN, PACS_PHOTO_DWT_STD, PACS_ART_DWT_MEAN, PACS_ART_DWT_STD, PACS_CARTOON_DWT_MEAN, PACS_CARTOON_DWT_STD, PACS_SKETCH_DWT_MEAN, PACS_SKETCH_DWT_STD, PACS_TOTAL_DWT_MEAN, PACS_TOTAL_DWT_STD 

# import cvt_functions as cCVT
_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50


class SwitchmageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
            corrupted=None,
            post_dwt=False,
            dataset_alias="imagenet",
            transform_sw=0,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        
        self.target_transform = target_transform
        
        self._consecutive_errors = 0
        self.corrupted=corrupted
        self.post_dwt=post_dwt
        self.dataset_alias=dataset_alias

        self.dwt_transform = list()
        self.dwt_level = 0
        
        self.primary_lst=None
        self.final_lst=None
        self.transform_switch=transform_sw 
        self.K = 0
        self.K_level = 0
        
    def __getitem__(self, index):
        img, target = self.reader[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0
        secondary_tfl = []
        if self.transform_switch == 0:
            if self.transform is not None:
                img = self.transform(img)
        elif self.transform_switch == 1:            
            secondary_tfl += auxiliary_argument_policy(0.5)
            aux_transform = T.Compose(self.primary_lst + secondary_tfl + self.final_lst)
            img = aux_transform(img)
        else:
            #Single K 
            assert 1 == 0

        if self.post_dwt == True:
            dwt_lst = list()

            img = img.unsqueeze(0)
            img *= 255
            tdwt.get_dwt_level1_CPU(img, dwt_lst)
            tdwt.make_figure_out(dwt_lst)

            img = torch.cat(dwt_lst,dim=0)
            img = img/255
            dwt_transform = self.normalize_transform()
            img = dwt_transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, [target, [0,0,0,0]]
    
    def normalize_transform(self):
        if self.dataset_alias == "imagenet":
                return T.Compose([
                    T.Normalize(mean=[x/255.0 for x in PACS_PHOTO_DWT_MEAN], std=[x/255.0 for x in PACS_PHOTO_DWT_STD]),
                ])
        elif self.dataset_alias == "photo":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_PHOTO_DWT_MEAN], std=[x/255.0 for x in PACS_PHOTO_DWT_STD]),
            ])
        elif self.dataset_alias == "art":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_ART_DWT_MEAN], std=[x/255.0 for x in PACS_ART_DWT_STD]),
            ])
        elif self.dataset_alias == "cartoon":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_CARTOON_DWT_MEAN], std=[x/255.0 for x in PACS_CARTOON_DWT_STD]),
            ])
        elif self.dataset_alias == "sketch":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_SKETCH_DWT_MEAN], std=[x/255.0 for x in PACS_SKETCH_DWT_STD]),
            ])
        elif self.dataset_alias == "pacs":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_TOTAL_DWT_MEAN], std=[x/255.0 for x in PACS_TOTAL_DWT_STD]),
            ])
        else:
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_PHOTO_DWT_MEAN], std=[x/255.0 for x in PACS_PHOTO_DWT_STD]),
            ])

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)
# Dataset Class For Meta Causal Learning
class MCImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
            corrupted=None,
            post_dwt=False,
            dataset_alias="imagenet",
            switch = [0,0,0,0] # sdm
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        print('@@@@@@@@@@@@@ MC Datasets')
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        
        self.target_transform = target_transform
        
        self._consecutive_errors = 0
        self.corrupted=corrupted
        self.post_dwt=post_dwt
        self.dataset_alias=dataset_alias

        self.dwt_transform = list()
        self.dwt_level = 0
        
        self.primary_lst=None
        self.final_lst=None
        self.transform_switch = switch[0]

        self.prob_1 = switch[1]
        self.prob_2 = switch[2]
        self.prob_3 = switch[3]

        self.K = 0
        self.K_level = 0

    # def K_setting(self, primary, final):
    #     self.primary_lst = primary
    #     self.final_lst = final

    #     policies = ['Brightness', 'Contrast', 'Color', 'Sharpness', 'Solarize', 
    #                 'SolarizeAdd', 'Posterize', 'SaltNoise', 'GaussianNoise', 'ShearX', 
    #                 'ShearY', 'Rotate']
    #     special_policies = ['AutoContrast', 'Invert', 'Equalize', 'FlipSingle']

    #     self.t_single = []
    #     for i in range(10):
    #         start_time = time.time()
    #         t_single = []
    #         second = auxiliary_argument_policy(0.5) 
    #         for policy in policies:
    #             for magnitude in [2, 3, 4, 6, 7]:
    #             #for magnitude in [2, 4, 7]:
    #             #for magnitude in [7]:
    #                 transform = T.Compose(self.primary_lst + second + auxliary_single_policy(policy, magnitude) + self.final_lst)
    #                 t_single.append(transform)

    #         for policy in special_policies:
    #             if policy == 'FlipSingle':
    #                 for magnitude in [1, 2, 3]:
    #                     transform = T.Compose(self.primary_lst + second + auxliary_single_policy(policy, magnitude) + self.final_lst)
    #                     t_single.append(transform)
    #             else:
    #                 transform = T.Compose(self.primary_lst + second + auxliary_single_policy(policy, 1) + self.final_lst)
    #                 t_single.append(transform)
    #         end = time.time()
    #         elapsed_time = end - start_time
    #         print(f"Elapsed time: {elapsed_time} seconds")
    #     self.t_single.append(t_single)
    
    # def apply_transform(self,tup):
    #     index, transform, img = tup
    #     print(f"Process {current_process().name} is transforming image with transform {index}")
    #     return index, transform(img)

    def __getitem__(self, index):
        img, target = self.reader[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0
        secondary_tfl = []
        third_tfl = []
        fourth_tfl = []
        if self.transform_switch == 0:
            if self.transform is not None:
                img = self.transform(img)
        elif self.transform_switch == 1:     
            secondary_tfl += auxiliary_argument_policy(0.75)
            aux_transform = T.Compose(self.primary_lst + secondary_tfl + self.final_lst)
            random_float = random.uniform(0, 1)
            if random_float > 0.5:
                img = self.transform(img)
            else:
                img = aux_transform(img) 
        elif self.transform_switch == 2: #2
            ori_img = self.transform(img)
            img_lst = [ori_img]
            if self.prob_1 > 0: 
                #print('prob_1 -> {}'.format(self.prob_1))
                secondary_tfl += auxiliary_argument_policy(self.prob_1)
                aux_transform = T.Compose(self.primary_lst + secondary_tfl + self.final_lst)
                img_lst.append(aux_transform(img))

            if self.prob_2 > 0: 
                #print('prob_2 -> {}'.format(self.prob_2))
                third_tfl += auxiliary_argument_policy(self.prob_2)
                aux_transform = T.Compose(self.primary_lst + third_tfl + self.final_lst)
                img_lst.append(aux_transform(img))

            if self.prob_3 > 0: 
                #print('prob_3 -> {}'.format(self.prob_3))
                fourth_tfl += auxiliary_argument_policy(self.prob_3)
                aux_transform = T.Compose(self.primary_lst + fourth_tfl + self.final_lst)
                img_lst.append(aux_transform(img))
                    
            img = torch.cat(img_lst, dim = 0)
        elif self.transform_switch == 3: #3
            ori_img = self.transform(img)
            img_lst = [ori_img]
            if self.prob_1 > 0: 
                secondary_tfl += fix_level_auxiliary_argument_policy(self.prob_1, 3)
                aux_transform = T.Compose(self.primary_lst + secondary_tfl + self.final_lst)
                img_lst.append(aux_transform(img))

            if self.prob_2 > 0: 
                third_tfl += fix_level_auxiliary_argument_policy(self.prob_2, 5)
                aux_transform = T.Compose(self.primary_lst + third_tfl + self.final_lst)
                img_lst.append(aux_transform(img))

            if self.prob_3 > 0: 
                fourth_tfl += fix_level_auxiliary_argument_policy(self.prob_3, 7)
                aux_transform = T.Compose(self.primary_lst + fourth_tfl + self.final_lst)
                img_lst.append(aux_transform(img))
                    
            img = torch.cat(img_lst, dim = 0)
        
        elif self.transform_switch == 4: #4
            ori_img = self.transform(img)
            img_lst = [ori_img]
            img_lst.append(auxliary_single_policy('Brightness',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('Contrast',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('Color',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('Sharpness',random.uniform(3.0, 7.0)))

            img_lst.append(auxliary_single_policy('AutoContrast',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('Invert',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('Equalize',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('Solarize',random.uniform(3.0, 7.0)))

            img_lst.append(auxliary_single_policy('SolarizeAdd',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('Posterize',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('SaltNoise',random.uniform(3.0, 7.0)))
            img_lst.append(auxliary_single_policy('GaussianNoise',random.uniform(3.0, 7.0)))

            for i in range(13):
                if i == 0:
                    continue
                seq = i
                aux = T.Compose(self.primary_lst + img_lst[seq] + self.final_lst)
                img_lst[seq] = aux(img)	
                    
            img = torch.cat(img_lst, dim = 0)
        elif self.transform_switch == 5: #5
            ori_img = self.transform(img)
            img_lst = [ori_img]
            img_lst.append(auxliary_single_policy('Brightness',5.0))
            img_lst.append(auxliary_single_policy('Contrast',5.0))
            img_lst.append(auxliary_single_policy('Color',5.0))
            img_lst.append(auxliary_single_policy('Sharpness',5.0))

            img_lst.append(auxliary_single_policy('AutoContrast',5.0))
            img_lst.append(auxliary_single_policy('Invert',5.0))
            img_lst.append(auxliary_single_policy('Equalize',5.0))
            img_lst.append(auxliary_single_policy('Solarize',5.0))

            img_lst.append(auxliary_single_policy('SolarizeAdd',5.0))
            img_lst.append(auxliary_single_policy('Posterize',5.0))
            img_lst.append(auxliary_single_policy('SaltNoise',5.0))
            img_lst.append(auxliary_single_policy('GaussianNoise',5.0))

            for i in range(13):
                if i == 0:
                    continue
                seq = i
                aux = T.Compose(self.primary_lst + img_lst[seq] + self.final_lst)
                img_lst[seq] = aux(img)	
                    
            img = torch.cat(img_lst, dim = 0)
        elif self.transform_switch == 6: #6: for validation
            ori_img = self.transform(img)
            img_lst = [ori_img]
            img_lst.append(auxliary_single_policy('Brightness', self.prob_1))
            img_lst.append(auxliary_single_policy('Contrast', self.prob_1))
            img_lst.append(auxliary_single_policy('Color', self.prob_1))
            img_lst.append(auxliary_single_policy('Sharpness', self.prob_1))

            img_lst.append(auxliary_single_policy('AutoContrast', self.prob_1))
            img_lst.append(auxliary_single_policy('Invert', self.prob_1))
            img_lst.append(auxliary_single_policy('Equalize', self.prob_1))
            img_lst.append(auxliary_single_policy('Solarize', self.prob_1))

            img_lst.append(auxliary_single_policy('SolarizeAdd', self.prob_1))
            img_lst.append(auxliary_single_policy('Posterize', self.prob_1))
            img_lst.append(auxliary_single_policy('SaltNoise', self.prob_1))
            img_lst.append(auxliary_single_policy('GaussianNoise', self.prob_1))

            for i in range(13):
                if i == 0:
                    continue
                seq = i
                aux = T.Compose(self.primary_lst + img_lst[seq] + self.final_lst)
                img_lst[seq] = aux(img)	
                    
            img = torch.cat(img_lst, dim = 0)
        else:
            ori_img = self.transform(img)
            img_count = int(self.prob_1)
            lst = []
            for i in range(img_count):
                lst.append(ori_img) 
            img = torch.cat(lst, dim=0)

        
        # else:
        #     ori_img = self.transform(img)
        #     img_count = int(self.prob_1)
        #     lst = []
        #     for i in range(img_count):
        #         lst.append(ori_img) 
        #     img = torch.cat(lst, dim=0)
            

        # else:
            # start = time.time()
            # if self.transform is not None:
            #     img_original = self.transform(img)
            
            # secondary_tfl += auxiliary_argument_policy(0.5)
            # aux_transform = T.Compose(self.primary_lst + secondary_tfl + self.final_lst)
            # img_aux = aux_transform(img)
            # elapsed_time = time.time()-start 
            # #print('original elapsed time -> {}'.format(elapsed_time))

            
            # #img_single_aux = [transform(img) for transform in self.t_single[0]]
            # # 결과를 저장할 리스트를 16개 만들어둡니다.
            # img_single_aux_list = []
            # start = time.time()
            # # 각 변환을 img에 적용하고 그 결과를 해당 리스트에 저장합니다.

            # with Pool(processes=16) as pool:
            #     results = pool.map(self.apply_transform, [(i, transform, img) for i, transform in enumerate(self.t_single[0])])
            #     for index, result in results:
            #         img_single_aux_list[index].append(result)

            # # for i, tr in enumerate(self.t_single[0]):
            # #     #print(tr)
            # #     start = time.time()
            # #     img_single_aux_list.append(tr(img))
            # #print('elapsed time -> {}'.format(elapsed_time))
            # # img_single_aux = [
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Brightness',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Brightness',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Brightness',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Brightness',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Brightness',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Contrast',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Contrast',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Contrast',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Contrast',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Contrast',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Color',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Color',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Color',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Color',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Color',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Sharpness',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Sharpness',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Sharpness',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Sharpness',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Sharpness',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Solarize',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Solarize',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Solarize',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Solarize',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Solarize',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SolarizeAdd',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SolarizeAdd',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SolarizeAdd',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SolarizeAdd',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SolarizeAdd',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Posterize',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Posterize',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Posterize',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Posterize',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Posterize',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SaltNoise',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SaltNoise',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SaltNoise',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SaltNoise',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('SaltNoise',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('GaussianNoise',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('GaussianNoise',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('GaussianNoise',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('GaussianNoise',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('GaussianNoise',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearX',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearX',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearX',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearX',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearX',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearY',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearY',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearY',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearY',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('ShearY',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Rotate',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Rotate',3) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Rotate',4) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Rotate',6) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Rotate',7) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('AutoContrast',1) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Invert',1) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('Equalize',1) +  self.final_lst)(img),

            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('FlipSingle',1) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('FlipSingle',2) +  self.final_lst)(img),
            # #     T.Compose(self.primary_lst + secondary_tfl + auxliary_single_policy('FlipSingle',3) +  self.final_lst)(img),
            # #     ]
            # elapsed_time = time.time()-start 
            # print('=> elapsed time -> {}'.format(elapsed_time))

        # if self.post_dwt == True:
        #     dwt_lst = list()

        #     img = img.unsqueeze(0)
        #     img *= 255
        #     tdwt.get_dwt_level1_CPU(img, dwt_lst)
        #     tdwt.make_figure_out(dwt_lst)

        #     img = torch.cat(dwt_lst,dim=0)
        #     img = img/255
        #     dwt_transform = self.normalize_transform()
        #     img = dwt_transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, [target, [0,0,0,0]] #+ img_single_aux
        
    def normalize_transform(self):
        if self.dataset_alias == "imagenet":
                return T.Compose([
                    T.Normalize(mean=[x/255.0 for x in PACS_PHOTO_DWT_MEAN], std=[x/255.0 for x in PACS_PHOTO_DWT_STD]),
                ])
        elif self.dataset_alias == "photo":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_PHOTO_DWT_MEAN], std=[x/255.0 for x in PACS_PHOTO_DWT_STD]),
            ])
        elif self.dataset_alias == "art":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_ART_DWT_MEAN], std=[x/255.0 for x in PACS_ART_DWT_STD]),
            ])
        elif self.dataset_alias == "cartoon":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_CARTOON_DWT_MEAN], std=[x/255.0 for x in PACS_CARTOON_DWT_STD]),
            ])
        elif self.dataset_alias == "sketch":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_SKETCH_DWT_MEAN], std=[x/255.0 for x in PACS_SKETCH_DWT_STD]),
            ])
        elif self.dataset_alias == "pacs":
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_TOTAL_DWT_MEAN], std=[x/255.0 for x in PACS_TOTAL_DWT_STD]),
            ])
        else:
            return T.Compose([
                T.Normalize(mean=[x/255.0 for x in PACS_PHOTO_DWT_MEAN], std=[x/255.0 for x in PACS_PHOTO_DWT_STD]),
            ])

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)

class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
            corrupted=None,
            post_dwt=False,
            dataset_alias="imagenet",
    ):
        print('Dataset Class => ImageDataset')
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.corrupted=corrupted
        self.post_dwt=post_dwt
        self.dataset_alias=dataset_alias

        self.dwt_transform = list()
        self.dwt_level = 0
        
    def __getitem__(self, index):
        img, target = self.reader[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.transform is not None:
            img = self.transform(img.convert('RGB'))
        
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        
        #print('ImageNet Dataset ')
        return img, [target, [0,0,0,0]]
        

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)

class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            is_training=False,
            batch_size=None,
            seed=42,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
