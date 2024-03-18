# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from copy import deepcopy

import torch
from torch.hub import load_state_dict_from_url
from utils.comm import is_main_process

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class VSTGCheckpointer(object):
    def __init__(
        self,
        cfg,
        model,
        model_ema=None,
        optimizer=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        is_train=True
    ):
        self.cfg = cfg
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.logger = logger
        self.is_train = is_train

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.model_ema is not None:
            data["model_ema"] = self.model_ema.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, with_optim=True, load_mapping={}):
        if self.has_checkpoint() and self.is_train:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from ImageNet")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        
        if with_optim:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        
        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)
    
    def _load_file(self, f):
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            self.logger.info("loading checking point from {}".format(f))
            loaded = load_state_dict_from_url(model_urls[self.cfg.MODEL.RESNETS.NAME])
        else:
            # load native pytorch checkpoint
            loaded = torch.load(f, map_location=torch.device("cpu"))

        return loaded
        
    def _load_mdetr_weight(self, weight_dict):
        load_mapping = {}
        current_keys = sorted(list(self.model.state_dict().keys()))

        for cur_key in current_keys:
            
            if cur_key.startswith('vis_encoder'):
                load_mapping[cur_key] = cur_key.replace('vis_encoder', 'backbone')
            
            if cur_key.startswith('text_encoder'):
                module_names = cur_key.split('.')
                if 'body' in module_names:
                    module_names.remove('body')
                else:
                    module_names.remove('text_encoder')
                    
                module_names.insert(0,'transformer')
                load_mapping[cur_key] = '.'.join(module_names)
                
            if  cur_key.startswith('input_proj'):
                load_mapping[cur_key] = cur_key
            
            if cur_key.startswith('bbox_embed'):
                load_mapping[cur_key] = cur_key
                    
            if cur_key.startswith('ground_encoder'):
                # ground_encoder.encoder.spatial_layers
                module_names = cur_key.split('.')
                if "spatial_layers" in module_names:
                    module_names.remove("ground_encoder")
                    module_names.insert(0,'transformer')
                    module_names.remove("spatial_layers")
                    module_names.insert(2,'layers')
                    load_mapping[cur_key] = '.'.join(module_names)

            if cur_key.startswith('ground_decoder'):
                module_names = cur_key.split('.')
                module_names.remove("ground_decoder")
                module_names.insert(0,'transformer')
                load_mapping[cur_key] = '.'.join(module_names)
                
        loaded_dict = {}
        for key in load_mapping:
            if load_mapping[key] in weight_dict.keys():
                loaded_dict[key] = weight_dict[load_mapping[key]]

        # for key in current_keys:
        #     if key not in loaded_dict.keys():
        #         print(key)

        self.model.load_state_dict(loaded_dict, strict=False)

    def _load_pretrained(self,state_dict):
        model_key = 'model'
        if "model_ema" in state_dict:
            model_key = 'model_ema'
        
        if self.is_train:
            # Initialized with the pretrained model weight
            self._load_mdetr_weight(state_dict[model_key])
            if 'args' in state_dict.keys():
                state_dict.pop('args')
            if 'epoch' in state_dict.keys():
                state_dict.pop('epoch')
            if 'optimizer' in state_dict.keys():
                state_dict.pop('optimizer')
        else:
            # Used For Evaluation and Inference, Load trained Checkpoint
            self.model.load_state_dict(state_dict[model_key])
        if (self.cfg.MODEL.EMA) and (self.model_ema is not None):
            self.model_ema.load_state_dict(deepcopy(self.model).state_dict()) 
   
    def _load_model(self, checkpoint):
        if self.is_train and self.has_checkpoint():   # resume training
            self.model.load_state_dict(checkpoint["model"])
            if (self.cfg.MODEL.EMA) and (self.model_ema is not None):
                if 'model_ema' not in checkpoint:
                    self.model_ema.load_state_dict(deepcopy(self.model).state_dict())
                else:
                    self.model_ema.load_state_dict(checkpoint["model_ema"])
        else:
            self._load_pretrained(checkpoint)
        if 'model_ema' in checkpoint:
            checkpoint.pop('model_ema')
        checkpoint.pop('model')