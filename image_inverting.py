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
from typing import Union


def register_attention_control(unet : nn.Module, controller:AttentionStore) :
    """ Register cross attention layers to controller. """
    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)

            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                                                         device=query.device),
                                             query,key.transpose(-1, -2),beta=0,alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            if not is_cross_attention:
                # when self attention
                query, key = controller.self_query_key_caching(query_value=query,
                                                               key_value=key,
                                                               layer_name=layer_name)
            else :
                query, key = controller.cross_query_key_caching(query_value=query,
                                                               key_value=key,
                                                               layer_name=layer_name)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states
        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count

    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
    controller.num_att_layers = cross_att_count

def unregister_attention_control(unet : nn.Module, controller:AttentionStore) :
    """ Register cross attention layers to controller. """
    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)

            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                                                         device=query.device),
                                             query,key.transpose(-1, -2),beta=0,alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states
        return forward
    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count

    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
    controller.num_att_layers = cross_att_count

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

    print(f" (1.0.1) logging")
    if args.log_with == 'wandb' :
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
    #text_encoder, vae, unet, _ = train_util._load_target_model(args, weight_dtype, accelerator)
    text_encoder, vae, unet, load_stable_diffusion_format = train_util._load_target_model(args,weight_dtype,args.device,
                                                                                          unet_use_linear_projection_in_v2=False,)
    model_version = model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization)
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    print(f' (1.3) register attention storer')
    attention_storer = AttentionStore()
    register_attention_control(unet, attention_storer)

    print(f' (1.4) scheduler')
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
    device = args.device
    if len(text_encoders) > 1:
        unet, t_enc1, t_enc2 = unet.to(device), text_encoders[0].to(device), text_encoders[1].to(device)
        text_encoder = text_encoders = [t_enc1, t_enc2]
        del t_enc1, t_enc2
    else:
        unet, text_encoder = unet.to(device), text_encoder.to(device)
        text_encoders = [text_encoder]

    print(f' \n step 2. groundtruth image preparing')
    print(f' (2.1) prompt condition')
    prompt = 'teddy bear, wearing sunglasses'
    def init_prompt(prompt: str):
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_input = tokenizer([prompt],padding="max_length",max_length=tokenizer.model_max_length,truncation=True,return_tensors="pt",)
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        context = torch.cat([uncond_embeddings, text_embeddings])
        return context
    context = init_prompt(prompt)
    print(f' (2.2) image condition')
    init_image_dir = '/data7/sooyeon/MyData/perfusion_dataset/td_100/100_td/td_1.jpg'
    image_gt_np = load_512(init_image_dir)

    def image2latent(image, vae, device):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device,weight_dtype)
                latents = vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    latent = image2latent(image_gt_np, vae, device)

    NUM_DDIM_STEPS = 50
    def call_unet(unet,noisy_latents, timesteps,text_conds, trg_indexs_list,mask_imgs):
        noise_pred = unet(noisy_latents,timesteps,text_conds,trg_indexs_list=trg_indexs_list,mask_imgs=mask_imgs, ).sample
        return noise_pred


    def next_step(model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
        alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    scheduler.set_timesteps(NUM_DDIM_STEPS)
    range_timesteps = len(scheduler.timesteps)
    print(f'len of scheduler timesteps : {range_timesteps}')
    @torch.no_grad()
    def ddim_loop(latent):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        all_latent = [latent]
        time_steps = []
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = scheduler.timesteps[len(scheduler.timesteps) - i - 1]
            time_steps.append(t)
            noise_pred = call_unet(unet, latent, t, cond_embeddings, None, None)
            latent = next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent, time_steps

    ddim_latents, time_steps = ddim_loop(latent)

    @torch.no_grad()
    def latent2image(latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    layer_names = attention_storer.self_query_store.keys()

    self_query_collection = attention_storer.self_query_store
    self_key_collection = attention_storer.self_key_store

    for layer in layer_names:
        cross_layer = layer.replace('attn1','attn2')
        self_query_list = attention_storer.self_query_store[layer]
        self_key_list = attention_storer.self_key_store[layer]
        cross_query_list = attention_storer.cross_query_store[cross_layer]
        cross_key_list = attention_storer.cross_key_store[cross_layer]
        i = 0
        for self_query, self_key, cross_query, cross_key in zip(self_query_list,self_key_list,cross_query_list,cross_key_list) :
            time_step = time_steps[i]
            print(f'time : {time_step} | layer_name : {layer} | self_query : {self_query.shape} | self_key : {self_key.shape}')
            print(f'time : {time_step} | cross_layer : {cross_layer} | cross_query : {cross_query.shape} | cross_key : {cross_key.shape}')
            i += 1
    """
    print(f' \n step 3. check latents')
    for i in range(len(ddim_latents)):
        trg_latent = ddim_latents[i]
        trg_img_np = latent2image(trg_latent)
        save_dir = os.path.join(args.output_dir, f'invert_{i}.jpg')
        os.makedirs(args.output_dir, exist_ok=True)
        Image.fromarray(trg_img_np).save(save_dir)
    """
    print(f' \n step 3. generating image')
    prompt = 'teddy bear, wearing sunglasses'
    prompt_list = [prompt]
    batch_size = len(prompt_list)
    height = width = 512
    text_input = context
    print(f'text_input (2,77,768) : {text_input.shape}')
    generator = None
    latent = torch.randn((1,unet.in_channels, height // 8, width // 8),
                         generator=generator,)
    latents = latent.expand(batch_size, unet.in_channels, height // 8, width // 8).to(device)
    print(f'latent : {latent.shape} | latents : {latents.shape}')

    start_time = 50
    guidance_scale = 7.5
    for i, t in enumerate(tqdm(scheduler.timesteps[-start_time:])):
        attention_storer.self_query_store = {}
        attention_storer.self_key_store = {}
        attention_storer.cross_query_store = {}
        attention_storer.cross_key_store = {}

        latents_input = torch.cat([latents] * 2)
        with torch.no_grad():
            noise_pred = unet(latents_input, t, encoder_hidden_states=context).sample
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

            trg_img_np = latent2image(latents)
            save_dir = os.path.join(args.output_dir, f'generating_{t.item()}.jpg')
            os.makedirs(args.output_dir, exist_ok=True)
            Image.fromarray(trg_img_np).save(save_dir)

    vae.to(device)
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        scheduler.config.clip_sample = True
    from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
    pipeline = StableDiffusionLongPromptWeightingPipeline(text_encoder=text_encoder, vae=vae, unet=unet, tokenizer=tokenizer,
                                                          scheduler=scheduler,
                                                          safety_checker=None, feature_extractor=None, requires_safety_checker=False,
                                                          clip_skip=args.clip_skip, )
    pipeline.to(device)
    prompt = 'teddy bear, wearing sunglasses'
    unregister_attention_control(unet, attention_storer)
    with torch.no_grad():
        negative_prompt =  'low quality, worst quality, bad anatomy,bad composition, poor, low effort'
        sample_steps = 30
        width = 512
        height = 512
        scale = 8
        seed = 42
        controlnet_image = None
        prompt = prompt
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8)  # round to divisible by 8
        latents = pipeline(prompt=prompt, height=height, width=width, num_inference_steps=sample_steps,
                           guidance_scale=scale, negative_prompt=negative_prompt,
                           controlnet_image=controlnet_image, )
        image = pipeline.latents_to_image(latents)[0]
        save_dir = os.path.join(args.output_dir, f'pipeline_gen.jpg')
        image.save(save_dir)

    """

        
        model.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale,
                                               low_resource=False)

        if return_type == 'image':
            image = ptp_utils.latent2image(model.vae, latents)
        else:
            image = latents
        return image, latent

    def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None,
                        verbose=True):
        if run_baseline:
            print("w.o. prompt-to-prompt")
            images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                             generator=generator)
            print("with prompt-to-prompt")
        images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                            num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                            generator=generator, uncond_embeddings=uncond_embeddings)
        if verbose:
            ptp_utils.view_images(images)
        return images, x_t
    """





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--device", type=str, default="cuda")
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