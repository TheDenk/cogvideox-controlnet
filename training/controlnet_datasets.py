import os
import glob
import random


import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from controlnet_aux import CannyDetector, HEDdetector


def unpack_mm_params(p):
    if isinstance(p, (tuple, list)):
        return p[0], p[1]
    elif isinstance(p, (int, float)):
        return p, p
    raise Exception(f'Unknown input parameter type.\nParameter: {p}.\nType: {type(p)}')


def resize_for_crop(image, min_h, min_w):
    img_h, img_w = image.shape[-2:]
    
    if img_h >= min_h and img_w >= min_w:
        coef = min(min_h / img_h, min_w / img_w)
    elif img_h <= min_h and img_w <=min_w:
        coef = max(min_h / img_h, min_w / img_w)
    else:
        coef = min_h / img_h if min_h > img_h else min_w / img_w 

    out_h, out_w = int(img_h * coef), int(img_w * coef)
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image


def init_controlnet(controlnet_type):
    if controlnet_type in ['canny']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators').to(device='cuda')


controlnet_mapping = {
    'canny': CannyDetector,
    'hed': HEDdetector,
}


class BaseClass(Dataset):
    def __init__(
            self, 
            video_root_dir,
            image_size=(320, 512), 
            stride=(1, 2), 
            sample_n_frames=25,
            hflip_p=0.5,
            controlnet_type='canny',
        ):
        self.height, self.width = unpack_mm_params(image_size)
        self.stride_min, self.stride_max = unpack_mm_params(stride)
        self.video_root_dir = video_root_dir
        self.sample_n_frames = sample_n_frames
        self.hflip_p = hflip_p
        
        self.length = 0
        
        self.controlnet_processor = init_controlnet(controlnet_type)
        
    def __len__(self):
        return self.length
        
    def load_video_info(self, video_path):
        video_reader = VideoReader(video_path)
        fps_original = video_reader.get_avg_fps()
        video_length = len(video_reader)
        
        sample_stride = random.randint(self.stride_min, self.stride_max)
        clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
        start_idx   = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        np_video = video_reader.get_batch(batch_index).asnumpy()
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 127.5 - 1
        del video_reader
        controlnet_video = [self.controlnet_processor(x) for x in np_video]
        controlnet_video = torch.from_numpy(np.stack(controlnet_video)).permute(0, 3, 1, 2).contiguous()
        controlnet_video = controlnet_video / 127.5 - 1
        return pixel_values, controlnet_video
        
    def get_batch(self, idx):
        raise Exception('Get batch method is not realized.')

    def __getitem__(self, idx):
        while True:
            try:
                video, caption, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        if self.hflip_p > random.random():
            video, controlnet_video = [
                transforms.functional.hflip(x) for x in [video, controlnet_video]
            ]
            
        video, controlnet_video = [
            resize_for_crop(x, self.height, self.width) for x in [video, controlnet_video]
        ] 
        video, controlnet_video = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [video, controlnet_video]
        ]
        data = {
            'video': video, 
            'caption': caption, 
            'controlnet_video': controlnet_video,
        }
        return data


class CustomControlnetDataset(BaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_paths = glob.glob(os.path.join(self.video_root_dir, '*.mp4'))
        self.length = len(self.video_paths)
        
    def get_batch(self, idx):
        video_path = self.video_paths[idx]
        caption = os.path.basename(video_path).replace('.mp4', '')
        pixel_values, controlnet_video = self.load_video_info(video_path)
        return pixel_values, caption, controlnet_video


class OpenvidControlnetDataset(BaseClass):
    def __init__(self, csv_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        videos_paths = glob.glob(os.path.join(self.video_root_dir, '*.mp4'))
        videos_names = set([os.path.basename(x) for x in videos_paths])
        self.df = pd.read_csv(csv_path)
        self.df['checked'] = self.df['path'].map(lambda x: int(x in videos_names))
        self.df = self.df[self.df['checked'] == True]
        self.length = self.df.shape[0]
        
    def get_batch(self, idx):
        item = self.df.iloc[idx]
        caption = item['text']
        video_name = item['path']
        video_path = os.path.join(self.video_root_dir, video_name)
        pixel_values, controlnet_video = self.load_video_info(video_path)
        return pixel_values, caption, controlnet_video
