"""
CLIP validation class

    Here we implement CLIP validation class for Text2Image and Image2Image similarity.
    
    We use CLIP model from OpenAI.(CLIP-ViT-B/32 at default, you can choose it by --clip_model argument)
"""

from library.validation_base import ValidationBase, ValidatorArgumentParserCallback
from argparse import ArgumentParser, Namespace
from transformers import CLIPModel, CLIPProcessor

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

import os
import glob
import re
import numpy as np
import torch


class CLIPArguments(ValidatorArgumentParserCallback):
    """
    Adds CLIP arguments to argparse.ArgumentParser.
    """
    def add_callback(parser: ArgumentParser):
        parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name.")
        parser.add_argument("--clip_device", type=str, default="cuda", help="CLIP device name.")
        parser.add_argument("--target_image_dir", type=str, default="dataset", help="Target dataset directory to load images.")
        # target text dir, which is captions for images, optional
        parser.add_argument("--target_text_dir", type=str, default=None, help="Target dataset directory to load texts.")
        # validation image dir, which is images to validate, which is usually samples from model
        parser.add_argument("--validation_image_dir", type=str, default="validation", help="Validation dataset directory to load images.")
        # regex to match from validation image dir, example : contains e000001 which means epoch 1 (6 digits filled with 0)
        parser.add_argument("--validation_image_regex", type=str, default="e{epoch:6d}", help="Regex to match validation images. default : e{epoch} which means epoch 1 (6 digits filled with 0)")
    
    

class CLIPValidation(ValidationBase):
    """
    CLIP validation class
    
    Here we implement CLIP validation class for Text2Image and Image2Image similarity.
    
    We use CLIP model from OpenAI.(openai/clip-vit-base-patch32 at default, you can choose it by --clip_model argument)
    """
    
    def __init__(self, clip_model:str="openai/clip-vit-base-patch32", device:str="cuda") -> None:
        """
        Initialize CLIP model.
        """
        self.clip_model = CLIPModel.from_pretrained(f"openai/{clip_model}").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(f"openai/{clip_model}")
        self.device = device
        self.cache = {} #'text' : {'name' : embedding}} 
        self.text_cache = {}
        self.target_image_cache = {}
        self.validation_image_cache = {}
        
    def start_validation(self, args:Namespace, **kwargs):
        """
        This method will be called at the start of validation.
        You can implement some initialization process here.
        This contains some loading cache, etc.
        """
        text_dir = args.target_text_dir
        caption_extension = args.caption_extension
        bool_text_worked = False
        for text_path in glob.glob(os.path.join(text_dir, "*" + caption_extension)):
            with open(text_path, "r", encoding='utf-8') as f:
                text = f.read()
            # remove extension
            self.text_cache[os.path.basename(text_path)[:-len(caption_extension)]] = text
            bool_text_worked = True
        assert bool_text_worked, "CLIPValidation text cache is empty. Something went wrong."
        # load target images
        bool_image_worked = False
        target_image_dir = args.target_image_dir
        for image_path in glob.glob(os.path.join(target_image_dir, "*")):
            # skip if it's not image, only accept png and jpg
            if not image_path.endswith(".png") and not image_path.endswith(".jpg"):
                continue
            self.target_image_cache[os.path.basename(image_path)] = image_path
            bool_image_worked = True
        assert bool_image_worked, "CLIPValidation target image cache is empty. Something went wrong."
        # load validation images
        regex_pattern = args.validation_image_regex
        current_epoch = kwargs.get("current_epoch", 0)
        current_step = kwargs.get("current_step", 0)
        # format regex pattern with current epoch
        if '{epoch' in regex_pattern:
            regex_pattern = regex_pattern.format(epoch=current_epoch)
        elif '{step' in regex_pattern:
            regex_pattern = regex_pattern.format(step=current_step)
        bool_validation_image_worked = False
        for image_path in glob.glob(os.path.join(args.validation_image_dir, "*")):
            # skip if it's not image, only accept png and jpg
            if not image_path.endswith(".png") and not image_path.endswith(".jpg"):
                continue
            # match regex
            if not re.match(regex_pattern, os.path.basename(image_path)):
                continue
            self.validation_image_cache[os.path.basename(image_path)] = image_path
            bool_validation_image_worked = True
        assert bool_validation_image_worked, "CLIPValidation validation image cache is empty. Something went wrong."
        # embed everything
        clip_text = {}
        for text_name, text in self.text_cache.items():
            clip_text[text_name] = self.clip_processor(text=text, return_tensors="pt").to(self.device)
        clip_image = {}
        for image_name, image_path in self.target_image_cache.items():
            clip_image[image_name] = self.clip_processor(images=image_path, return_tensors="pt").to(self.device)
        clip_validation_image = {}
        for image_name, image_path in self.validation_image_cache.items():
            clip_validation_image[image_name] = self.clip_processor(images=image_path, return_tensors="pt").to(self.device)
        self.cache.update({"text" : clip_text, "image" : clip_image, "validation_image" : clip_validation_image})
        assert self.cache != {} and self.text_cache != {} and self.target_image_cache != {} and self.validation_image_cache != {}, "CLIPValidation cache is empty. Something went wrong."
        # it is now prepared. call log next step
        
    def end_validation(self, **kwargs):
        """
        This method will be called at the end of validation.
        You can implement some finalization process here.
        This contains some cleaning up cache, etc.
        """
        self.cache = {}
        self.text_cache = {}
        self.target_image_cache = {}
        self.validation_image_cache = {}
        
    def log(self, args:Namespace, log_dict:dict, **kwargs) -> dict:
        """
        Log the score from args.
        It will append the score to log_dict then return it.
        """
        # now we will calculate text-validation_image similarity and image-validation_image similarity
        embed_text = self.cache["text"]
        embed_image = self.cache["image"]
        embed_validation_image = self.cache["validation_image"]
        # calculate text-validation_image similarity
        tensor_text = torch.cat([embed_text[text_name]["input_ids"] for text_name in embed_text.keys()], dim=0)
        tensor_validation_image = torch.cat([embed_validation_image[image_name]["pixel_values"] for image_name in embed_validation_image.keys()], dim=0)
        tensor_image = torch.cat([embed_image[image_name]["pixel_values"] for image_name in embed_image.keys()], dim=0)
        # now as array, get similarity
        similarity_text_validation_image = cosine_similarity(tensor_text.cpu().numpy(), tensor_validation_image.cpu().numpy())
        similarity_image_validation_image = cosine_similarity(tensor_image.cpu().numpy(), tensor_validation_image.cpu().numpy())
        # get mean and std
        mean_text_validation_image = np.mean(similarity_text_validation_image)
        std_text_validation_image = np.std(similarity_text_validation_image)
        mean_image_validation_image = np.mean(similarity_image_validation_image)
        std_image_validation_image = np.std(similarity_image_validation_image)
        # update log_dict
        dict_to_add = {"val/score_text_validation_image_mean" : mean_text_validation_image, "val/score_text_validation_image_std" : std_text_validation_image, "val/score_image_validation_image_mean" : mean_image_validation_image, "val/score_image_validation_image_std" : std_image_validation_image}
        log_dict.update(dict_to_add)
        return log_dict
    
    
# register
CustomValidatorArgumentParserCallback = CLIPArguments
CustomValidator = CLIPValidation
