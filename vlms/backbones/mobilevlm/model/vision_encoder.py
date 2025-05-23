import os
import torch
import random
import torch.nn as nn
from configs.vision_encoders import encoder_name_to_path
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        print(self.vision_tower_name)
        self.vision_tower_path = encoder_name_to_path.get(self.vision_tower_name, self.vision_tower_name)
        print(self.vision_tower_path)
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        if not delay_load:
            self.load_model() 
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_path)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_path)  # dummy-load

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_path)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def load_image_processor(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path)
        self.is_loaded = True
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        if self.select_feature.startswith('mtcv'):
            num_select = int(self.select_feature.split('-')[-1])
            return self.config.hidden_size * num_select
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def build_vision_tower(model_cfg, **kwargs):
    vision_tower = getattr(model_cfg, 'mm_vision_tower', getattr(model_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        vision_tower_type = getattr(model_cfg, 'vision_tower_type', None)
        if vision_tower_type == "clip":
            return CLIPVisionTower(vision_tower, args=model_cfg, **kwargs)
    raise ValueError(f'Unknown vision tower: {vision_tower}')