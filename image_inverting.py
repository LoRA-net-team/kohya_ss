# get image vector with semantic from each time step and spatial layer block position
import importlib
import argparse
import gc
import re
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
from tqdm import tqdm
import toml
import tempfile
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import model_util
import library.train_util as train_util
from library.train_util import (DreamBoothDataset,)
import library.config_util as config_util
from library.config_util import (ConfigSanitizer,BlueprintGenerator,)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (apply_snr_weight, get_weighted_text_embeddings,prepare_scheduler_for_custom_training,
                                            scale_v_prediction_loss_like_noise_prediction,add_v_prediction_like_loss,)
import torch
from torch import nn
import torch.nn.functional as F
from functools import lru_cache
from attention_store import AttentionStore
import wandb
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
from diffusers import (StableDiffusionPipeline,DDPMScheduler,EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler,
                       DPMSolverSinglestepScheduler,LMSDiscreteScheduler,PNDMScheduler,DDIMScheduler,
                       EulerDiscreteScheduler,HeunDiscreteScheduler,KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler,
                       AutoencoderKL,)
import numpy as np
from PIL import Image

"""
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer
"""

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def image2latent(image, vae, device):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            latents = vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents

def main(args) :

    print(f' \n step 1. make stable diffusion model')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')

    session_id = random.randint(0, 2 ** 32)
    training_started_at = time.time()
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None
    use_user_config = args.dataset_config is not None
    use_class_caption = args.class_caption is not None  # if class_caption is provided, for subsets, add key 'class_caption' to each subset

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f" (1.0.1) preparing accelerator")
    accelerator = train_util.prepare_accelerator(args)
    is_main_process = accelerator.is_main_process
    if args.log_with == 'wandb' and is_main_process:
        wandb.init(project=args.wandb_init_name, name=args.wandb_run_name)

    print(f" (1.0.2) save directory and save config")
    save_base_dir = args.output_dir
    _, folder_name = os.path.split(save_base_dir)
    record_save_dir = os.path.join(args.output_dir, "record")
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f" (1.0.3) save directory and save config")
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    print(f' (1.1) tokenizer')
    tokenizer = train_util.load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

    print(f' (1.2) SD')
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
    model_version = model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization)
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    
    print(f' (1.3) scheduler')
    sched_init_args = {}
    if args.sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif args.sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif args.sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif args.sample_sampler == "lms" or args.sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif args.sample_sampler == "euler" or args.sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_sampler == "euler_a" or args.sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_sampler == "dpmsolver" or args.sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sample_sampler
    elif args.sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_sampler == "dpm_2" or args.sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif args.sample_sampler == "dpm_2_a" or args.sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler
    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction "

    # scheduler:
    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"
    scheduler = scheduler_cls(num_train_timesteps=SCHEDULER_TIMESTEPS,
                              beta_start=SCHEDULER_LINEAR_START,
                              beta_end=SCHEDULER_LINEAR_END,
                              beta_schedule=SCHEDLER_SCHEDULE,)

    print(f' (1.4) model to accelerator device')
    if len(text_encoders) > 1:
        unet, t_enc1, t_enc2, = accelerator.prepare(unet, text_encoders[0], text_encoders[1])
        text_encoder = text_encoders = [t_enc1, t_enc2]
        del t_enc1, t_enc2
    else:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        text_encoders = [text_encoder]

    print(f' \n step 2. groundtruth image preparing')
    print(f' (2.1) prompt condition')
    prompt = 'teddy bear, wearing sunglasses'
    def init_prompt(prompt: str):
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[0]
        text_input = tokenizer([prompt],padding="max_length",max_length=tokenizer.model_max_length,truncation=True,return_tensors="pt",)
        text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
        context = torch.cat([uncond_embeddings, text_embeddings])
        return context
    context = init_prompt(prompt)
    print(f' (2.2) image condition')
    init_image_dir = '/data7/sooyeon/MyData/perfusion_dataset/td_100/100_td/td_1.jpg'
    image_gt_np = load_512(init_image_dir)
    latent = image2latent(image_gt_np, vae, accelerator.device)
    """
    @torch.no_grad()
    def ddim_loop(latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = scheduler.timesteps[len(scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent


    ddim_latents = ddim_loop(latent)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--no_metadata", action="store_true",
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）", )
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=None,
                        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）")
    parser.add_argument("--network_alpha",type=float,default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)",)
    parser.add_argument("--network_dropout",type=float,default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)",)
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value) / ネットワークへの追加の引数")
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--dim_from_weights",action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",)
    parser.add_argument("--scale_weight_norms",type=float,default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ",)
    parser.add_argument("--base_weights",type=str,default=None,nargs="*",
                        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",)
    parser.add_argument("--base_weights_multiplier",type=float,default=None,nargs="*",
                        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",)
    parser.add_argument("--no_half_vae",action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",)
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    parser.add_argument("--wandb_init_name", type=str)
    parser.add_argument("--wandb_log_template_path", type=str)
    parser.add_argument("--wandb_key", type=str)
    parser.add_argument("--trg_concept", type=str, default='haibara')
    parser.add_argument("--net_key_names", type=str, default='text')

    # class_caption
    parser.add_argument("--class_caption", type=str, default='girl')
    parser.add_argument("--heatmap_loss", action='store_true')
    parser.add_argument("--attn_loss_ratio", type=float, default=1.0)
    parser.add_argument("--mask_dir", type=str)

    # masked_loss
    parser.add_argument("--masked_loss", action='store_true')
    parser.add_argument("--only_second_training", action='store_true')
    parser.add_argument("--only_third_training", action='store_true')
    parser.add_argument("--first_second_training", action='store_true')
    parser.add_argument("--second_third_training", action='store_true')
    parser.add_argument("--first_second_third_training", action='store_true')
    parser.add_argument("--attn_loss_layers", type=str, default="all", help="attn loss layers, can be splitted with ',', matches regex with given string. default is 'all'")
    # mask_threshold (0~1, default 1)
    parser.add_argument("--mask_threshold", type=float, default=1.0, help="Threshold for mask to be used as 1")
    parser.add_argument("--heatmap_backprop", action = 'store_true')
    args = parser.parse_args()

    args = train_util.read_config_from_file(args, parser)
    main(args)