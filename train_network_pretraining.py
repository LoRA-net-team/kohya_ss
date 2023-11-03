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
@lru_cache(maxsize=128)
def match_layer_name(layer_name:str, regex_list_str:str) -> bool:
    """
    Check if layer_name matches regex_list_str.
    """
    regex_list = regex_list_str.split(',')
    for regex in regex_list:
        regex = regex.strip() # remove space
        if re.match(regex, layer_name):
            return True
    return False
def register_attention_control(unet : nn.Module, controller:AttentionStore, mask_threshold:float=1): #if mask_threshold is 1, use itself
    """
    Register cross attention layers to controller.
    """
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

            if is_cross_attention:
                if trg_indexs_list is not None and mask is not None:
                    trg_indexs = trg_indexs_list
                    batch_num = len(trg_indexs)
                    attention_probs_batch = torch.chunk(attention_probs, batch_num, dim=0)
                    for batch_idx, attention_prob in enumerate(attention_probs_batch) :
                        batch_trg_index = trg_indexs[batch_idx] # two times
                        head_num = attention_prob.shape[0]
                        res = int(math.sqrt(attention_prob.shape[1]))
                        word_heat_map_list = []
                        for word_idx in batch_trg_index :
                            # head, pix_len
                            word_heat_map = attention_prob[:, :, word_idx]
                            word_heat_map_ = word_heat_map.reshape(-1, res, res)
                            word_heat_map_ = word_heat_map_.mean(dim=0)
                            word_heat_map_ = F.interpolate(word_heat_map_.unsqueeze(0).unsqueeze(0),
                                                           size=((512, 512)),mode='bicubic').squeeze()
                            word_heat_map_list.append(word_heat_map_)
                        word_heat_map_ = torch.stack(word_heat_map_list, dim=0) # (word_num, 512, 512)
                        # saving word_heat_map
                        # ------------------------------------------------------------------------------------------------------------------------------
                        # mask = [512,512]
                        mask_ = mask[batch_idx].to(attention_prob.dtype) # (512,512)
                        # thresholding, convert to 1 if upper than threshold else itself
                        mask_ = torch.where(mask_ > mask_threshold, torch.ones_like(mask_), mask_)
                        # check if mask_ is frozen, it should not be updated
                        assert mask_.requires_grad == False, 'mask_ should not be updated'
                        masked_heat_map = word_heat_map_ * mask_
                        attn_loss = F.mse_loss(word_heat_map_.mean(), masked_heat_map.mean())
                        controller.store(attn_loss, layer_name)
                # check if torch.no_grad() is in effect
                elif torch.is_grad_enabled(): # if not, while training, trg_indexs_list should not be None
                    if mask is None:
                        raise RuntimeError("mask is None but hooked to cross attention layer. Maybe the dataset does not contain mask properly.")
                    raise RuntimeError("trg_indexs_list is None but hooked to cross attention layer. Maybe the dataset does not contain trigger token properly.")

            hidden_states = torch.bmm(attention_probs, value)
            #if is_cross_attention :
            #    print(f'layer {layer_name} hidden_states.shape : {hidden_states.shape}')
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

def arg_as_list(s):
    import ast
    v = ast.literal_eval(s)
    return v

class NetworkTrainer:

    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(self, args: argparse.Namespace, current_loss, avr_loss, lr_scheduler,
                           keys_scaled=None, mean_norm=None, maximum_norm=None, **kwargs):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}
        # ------------------------------------------------------------------------------------------------------------------------------
        # updating kwargs with new loss logs ...
        if kwargs is not None:
            logs.update(kwargs)
        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm
        lrs = lr_scheduler.get_last_lr()
        if args.network_train_text_encoder_only or len(lrs) <= 2:  # not block lr (or single block)
            if args.network_train_unet_only:
                logs["lr/unet"] = float(lrs[0])
            elif args.network_train_text_encoder_only:
                logs["lr/textencoder"] = float(lrs[0])
            else:
                logs["lr/textencoder"] = float(lrs[0])
                logs["lr/unet"] = float(lrs[-1])  # may be same to textencoder

            if (
                args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                    lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                )
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )

        return logs

    def assert_extra_args(self, args, train_dataset_group):
        pass

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_tokenizer(self, args):
        tokenizer = train_util.load_tokenizer(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return False
    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype ):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device)

    def extract_triggerword_index(self, input_ids):
        cls_token = 49406
        pad_token = 49407
        if input_ids.dim() == 3 :
            input_ids = torch.flatten(input_ids, start_dim=1)
        batch_num, sen_len = input_ids.size()
        batch_index_list = []

        for batch_index in range(batch_num) :
            token_ids = input_ids[batch_index, :].squeeze()
            index_list = []
            for index, token_id in enumerate(token_ids):
                if token_id != cls_token and token_id != pad_token :
                    index_list.append(index)
            batch_index_list.append(index_list)
        return batch_index_list

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids,tokenizers[0], text_encoders[0],weight_dtype )
        return encoder_hidden_states

    def call_unet(self,args, accelerator, unet,
                  noisy_latents, timesteps,
                  text_conds, batch, weight_dtype,
                  trg_indexs_list,
                  mask_imgs):
        noise_pred = unet(noisy_latents,
                          timesteps,
                          text_conds,
                          trg_indexs_list=trg_indexs_list,
                          mask_imgs=mask_imgs, ).sample
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet
                      ,efficient=False):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet,
                                 efficient=efficient)

    def get_input_ids(self, args, caption, tokenizer):
        tokenizer_max_length = args.max_token_length + 2
        # [1,77]
        input_ids = tokenizer(caption, padding="max_length", truncation=True, max_length=tokenizer_max_length, return_tensors="pt").input_ids
        if tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                for i in range( 1, tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2) :
                    ids_chunk = (input_ids[0].unsqueeze(0),input_ids[i : i + tokenizer.model_max_length - 2],  input_ids[-1].unsqueeze(0),)
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                for i in range(1, tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id
                    iids_list.append(ids_chunk)
            # [3,77]
            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids






    def train(self, args):

        print("\n step 1. start session")
        if args.process_title :
            setproctitle(args.process_title)
        else :
            setproctitle('parksooyeon')

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None
        use_class_caption = args.class_caption is not None # if class_caption is provided, for subsets, add key 'class_caption' to each subset

        print("\n step 2. seed")
        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        print("\n step 3. preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process
        if args.log_with == 'wandb' and is_main_process:
            import wandb
            wandb.init(project=args.wandb_init_name,name=args.wandb_run_name)

        print("\n step 4. save config")
        save_base_dir = args.output_dir
        _, folder_name = os.path.split(save_base_dir)
        record_save_dir = os.path.join(args.output_dir, "record")
        os.makedirs(record_save_dir, exist_ok=True)
        with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        print("\n step 5. loading stable diffusion")
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        _, text_encoder_org, vae_org, unet_org = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders_org = text_encoder_org if isinstance(text_encoder_org, list) else [text_encoder_org]


        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        print("\n step 6. Dataset & Loader 1")
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
            if use_user_config:
                print(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    print("ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(", ".join(ignored)))
            else:
                if use_dreambooth_method:
                    print("Using DreamBooth method.")
                    user_config = {}
                    user_config['datasets'] = [{"subsets" : None}]
                    subsets_dict_list = []
                    for subsets_dict in config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir):
                        if use_class_caption:
                            subsets_dict['class_caption'] = args.class_caption
                        subsets_dict_list.append(subsets_dict)
                        user_config['datasets'][0]['subsets'] = subsets_dict_list
                else:
                    print("Training with captions.")
                    user_config = {}
                    user_config["datasets"] = []
                    user_config["datasets"].append({"subsets": [{"image_dir": args.train_data_dir,"metadata_file": args.in_json,}]})
                    if use_class_caption:
                        for subset in user_config["datasets"][0]["subsets"]:
                            subset["class_caption"] = args.class_caption
            blueprint = blueprint_generator.generate(user_config,args,tokenizer=tokenizer)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)
        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            print("No data found. Please verify arguments (train_data_dir must be the parent of folders with images) ）")
            return
        if cache_latents:
            assert (train_dataset_group.is_latent_cacheable()), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        self.assert_extra_args(args, train_dataset_group)
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで
        train_dataloader = torch.utils.data.DataLoader(train_dataset_group, batch_size=1, shuffle=True, collate_fn=collater,
                                                       num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers, )
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        print("\n step 6. Dataset & Loader 2")
        class_caption_dir = args.class_caption_dir
        with open(class_caption_dir, 'r') as f:
            class_captions = f.readlines()
        class_captions = [caption.strip() for caption in class_captions]
        class_captions = [caption.lower() for caption in class_captions]

        class TE_dataset(torch.utils.data.Dataset):
            def __init__(self, class_captions):
                class_token = args.class_token
                trg_concept = args.trg_concept
                self.class_captions = class_captions
                self.concept_captions = [caption.replace(class_token, trg_concept) for caption in class_captions]
            def __len__(self):
                return len(self.class_captions)
            def __getitem__(self, idx):
                te_example = {}
                # 1) class
                class_caption = self.class_captions[idx]
                # 2) concept
                concept_caption = self.concept_captions[idx]
                # 3) total
                te_example['class_caption'] = class_caption
                te_example['concept_caption'] = concept_caption
                return te_example
        pretraining_datset = TE_dataset(class_captions=class_captions)
        pretraining_dataloader = torch.utils.data.DataLoader(pretraining_datset, batch_size=1, shuffle=True,
                                                             num_workers=n_workers,
                                                             persistent_workers=args.persistent_data_loader_workers, )

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        print("\n step 7-1. prepare network")
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)
        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]
                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")
                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True             )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")
            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
            vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            accelerator.wait_for_everyone()
        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, tokenizers, text_encoders, train_dataset_group, weight_dtype)
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value
        net_key_names = args.net_key_names
        net_kwargs['key_layers'] = net_key_names.split(",")
        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            # LyCORIS will work with this...
            network = network_module.create_network(1.0, args.network_dim, args.network_alpha,
                                                    vae, text_encoder, unet,
                                                    neuron_dropout=args.network_dropout,
                                                    **net_kwargs,)
        if network is None:
            return
        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            print("warning: scale_weight_norms is specified but the network does not support it / scale_weight_norms")
            args.scale_weight_norms = False
        train_unet = False
        train_text_encoder = not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        print("\n step 8-1. optimizer (with only text encoder loras)")
        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except TypeError:
            accelerator.print(
                "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)")
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        print("\n step 9-1. learning rate")
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
        if args.full_fp16:
            assert (args.mixed_precision == "fp16"), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (args.mixed_precision == "bf16"), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)
        unet.requires_grad_(False)
        unet.to(dtype=weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)
        print("\n step 11-1. module to accelerate")
        if len(text_encoders) > 1:
            t_enc1, t_enc2, t_org_enc1, t_org_enc2, network, optimizer, pretraining_dataloader, lr_scheduler = accelerator.prepare(text_encoders[0], text_encoders[1],
                                                                                                                                   text_encoders_org[0],text_encoders_org[1],
                                                                                                                                   network,optimizer,pretraining_dataloader,
                                                                                                                                   lr_scheduler)
            text_encoder = text_encoders = [t_enc1, t_enc2]
            text_encoder_org = text_encoders_org = [t_org_enc1,t_org_enc2]
            del t_enc1, t_enc2, t_org_enc1,t_org_enc2
        else:
            text_encoder, text_encoder_org, network, optimizer, pretraining_dataloader, lr_scheduler = accelerator.prepare(text_encoder, text_encoder_org,
                                                                                                                           network, optimizer, pretraining_dataloader, lr_scheduler)
            text_encoders = [text_encoder]
            text_encoders_org = [text_encoder_org]

        text_encoders = train_util.transform_models_if_DDP(text_encoders)
        text_encoders_org = train_util.transform_models_if_DDP(text_encoders_org)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        print("\n step 12. text encoder pretraining")
        if is_main_process :
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_name = f'{args.output_name}-base.safetensors'
            ckpt_file = os.path.join(args.output_dir, ckpt_name)
            unwrapped_nw = accelerator.unwrap_model(network)
            unwrapped_nw.save_weights(ckpt_file, save_dtype,metadata=None)

        print("\n step 13. image inference before training anything")
        #self.sample_images(accelerator, args, -1, 0, accelerator.device, vae, tokenizer, text_encoder, unet)

        print("\n step 14. text encoder lora pretraining")
        pretraining_epochs = args.pretraining_epochs
        pretraining_losses = {}
        for epoch in range(pretraining_epochs):
            for batch in pretraining_dataloader:
                class_caption = batch['class_caption']
                class_token_ids = self.get_input_ids(args, class_caption, tokenizer).unsqueeze(0)
                class_captions_hidden_states = train_util.get_hidden_states(args,
                                                                            class_token_ids.to(accelerator.device),
                                                                            tokenizers[0], text_encoders_org[0],
                                                                            weight_dtype)
                # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                #class_captions_lora_hidden_states = train_util.get_hidden_states(args,class_token_ids.to(accelerator.device),
                #                                                                   tokenizers[0], text_encoders[0],
                #                                                                   weight_dtype)
                concept_caption = batch['concept_caption']
                concept_captions_input_ids = self.get_input_ids(args, concept_caption, tokenizer).unsqueeze(0)
                concept_captions_lora_hidden_states = train_util.get_hidden_states(args,concept_captions_input_ids.to(accelerator.device),
                                                                                   tokenizers[0], text_encoders[0],
                                                                                   weight_dtype)
                # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # shape = [3,77,768]
                #preservating_loss = torch.nn.functional.mse_loss(class_captions_hidden_states.float(),
                #                                                 class_captions_lora_hidden_states.float(),
                #                                                 reduction="none")
                pretraining_loss = torch.nn.functional.mse_loss(class_captions_hidden_states.float(),
                                                                concept_captions_lora_hidden_states.float(),
                                                                reduction="none")
                #pretraining_losses["loss/preservating_loss"] = preservating_loss.mean().item()
                pretraining_losses["loss/pretraining_loss"] = pretraining_loss.mean().item()
                if is_main_process:
                    # accelerator.log(pretraining_losses)
                    wandb.log(pretraining_losses)
                #te_loss = preservating_loss.mean() + pretraining_loss.mean()
                te_loss = pretraining_loss.mean()
                accelerator.backward(te_loss)
                optimizer.step()
                lr_scheduler.step()

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        print("\n step 7-2. prepare unet network")
        network = accelerator.unwrap_model(network)
        unet_key_names = args.unet_net_key_names.split(",")
        network.add_unet_module(unet, net_key_names=unet_key_names)
        network.apply_unet_to(apply_unet=True, )
        print("\n step 8-2. optimizer (with only text encoder loras)")
        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except TypeError:
            accelerator.print(
                "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)")
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

        if args.te_freeze :
            trainable_params = [trainable_params[-1]]

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
        print("\n step 9-2. learning rate")
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
        if args.full_fp16:
            assert (args.mixed_precision == "fp16"), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (args.mixed_precision == "bf16"), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        print("\n step 10-2. train epochs")
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")
        train_dataset_group.set_max_train_steps(args.max_train_steps)
        unet.requires_grad_(False)
        unet.to(dtype=weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)
        print("\n step 11-2. module to accelerate")
        if len(text_encoders) > 1:
            unet, t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler)
            text_encoder = text_encoders = [t_enc1, t_enc2]
            del t_enc1, t_enc2
        else:
            unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler)
            text_encoders = [text_encoder]
        text_encoders = train_util.transform_models_if_DDP(text_encoders)
        unet, network = train_util.transform_models_if_DDP([unet, network])


        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()
                # set top parameter requires_grad = True for gradient checkpointing works
                if train_text_encoder:
                    t_enc.text_model.embeddings.requires_grad_(True)
            # set top parameter requires_grad = True for gradient checkpointing works
            if not train_text_encoder:  # train U-Net only
                unet.parameters().__next__().requires_grad_(True)
        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()
        del t_enc
        network.prepare_grad_etc(text_encoder, unet)

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
        if args.heatmap_loss:
            attention_storer = AttentionStore()
            register_attention_control(unet, attention_storer, mask_threshold=args.mask_threshold)
        else:
            attention_storer = None

        # -----------------------------------------------------------------------------------------------------------------
        # effective sampling
        text_encoder_org = accelerator.unwrap_model(text_encoder_org)




        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
        }

        if use_user_config:
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,

                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"
            dataset = train_dataset_group.datasets[0]
            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats,
                                                                "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,}
            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),})

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name
        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name
        metadata = {k: str(v) for k, v in metadata.items()}
        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
        global_step = 0
        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=1000, clip_sample=False)
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers("network_train" if args.log_tracker_name is None else args.log_tracker_name,
                                      init_kwargs=init_kwargs)

        loss_list = []
        loss_total = 0.0
        del train_dataset_group
        # callback for step start
        if hasattr(network, "on_step_start"):
            on_step_start = network.on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        # function for saving/removing
        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)
        # save base model
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)
            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        if is_main_process :
            ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, 0)
            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, 0)
        #self.sample_images(accelerator, args, 0, 0, accelerator.device, vae, tokenizer,text_encoder, unet)

        print("\n step 13. training loop")
        attn_loss_records = [['epoch', 'global_step', 'attn_loss']]
        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1
            metadata["ss_epoch"] = str(epoch + 1)
            network.on_epoch_start(text_encoder, unet)
            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                with accelerator.accumulate(network):
                    on_step_start(text_encoder, unet)
                    with torch.no_grad():
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(accelerator.device)
                        else:
                            latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample()
                            # NaNが含まれていれば警告を表示し0に置き換える
                            if torch.any(torch.isnan(latents)):
                                accelerator.print("NaN found in latents, replacing with zeros")
                                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                        latents = latents * self.vae_scale_factor
                    b_size = latents.shape[0]
                    with torch.set_grad_enabled(train_text_encoder):
                        # Get the text embedding for conditioning
                        if args.weighted_captions:
                            text_encoder_conds = get_weighted_text_embeddings(tokenizer,text_encoder,
                                                                              batch["captions"],accelerator.device,
                                                                              args.max_token_length // 75 if args.max_token_length else 1,
                                                                              clip_skip=args.clip_skip,)
                        else:
                            text_encoder_conds = self.get_text_cond(args, accelerator, batch, tokenizers, text_encoders, weight_dtype)
                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
                    # Predict the noise residual
                    with accelerator.autocast():
                        noise_pred = self.call_unet(args,accelerator,unet,noisy_latents,timesteps,text_encoder_conds,
                                                    batch,weight_dtype,
                                                    batch["trg_indexs_list"],
                                                    #trg_index_list,
                                                    batch['mask_imgs'])
                        if attention_storer is not None:
                            atten_collection = attention_storer.step_store
                            attention_storer.step_store = {}


                    if args.v_parameterization:
                        # v-parameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise


                    ### Masked loss ###
                    if args.masked_loss:
                        mask_imgs = [mask_img.unsqueeze(0).unsqueeze(0) for mask_img in batch['mask_imgs']]
                        # interpolate
                        mask_imgs = [F.interpolate(mask_img, noise_pred.size()[-2:], mode='bilinear') for mask_img in mask_imgs]
                        # to Tensor
                        mask_imgs = torch.cat(mask_imgs, dim=0) # [batch_size, 1, 256, 256]
                        #print("mask_imgs size: ", mask_imgs[0].size()) # debug
                        # multiply mask to noise_pred and target
                        # element-wise multiplication
                        noise_pred = noise_pred * mask_imgs
                        target = target * mask_imgs
                    batch_num = noise_pred.shape[0]
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])
                    loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                    loss = loss * loss_weights
                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし
                    task_loss = loss
                    attn_loss = 0
                    attention_losses = {}
                    attention_losses["loss/task_loss"] = loss

                    # ------------------------------------------------------------------------------------
                    if args.heatmap_loss:
                        layer_names = atten_collection.keys()
                        assert len(
                            layer_names) > 0, "Cannot find any layer names in attention_storer. check your model."
                        heatmap_per_batch = {}
                        for layer_name in layer_names:
                            if args.attn_loss_layers == 'all' or match_layer_name(layer_name, args.attn_loss_layers):
                                sum_of_attn = sum(atten_collection[layer_name])
                                attn_loss = attn_loss + sum_of_attn
                                attention_losses["loss/attention_loss_" + layer_name] = sum_of_attn
                        attention_losses["loss/attention_loss"] = attn_loss
                        assert attn_loss != 0, f"attn_loss is 0. check attn_loss_layers or attn_loss_ratio.\n available layers: {layer_names}\n given layers: {args.attn_loss_layers}"
                        if args.heatmap_backprop:
                            loss = task_loss + args.attn_loss_ratio * attn_loss
                    else:
                        attention_losses = {}
                    accelerator.backward(loss)

                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = network.get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = network.apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device)
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm }
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    #self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)
                    if attention_storer is not None:
                        attention_storer.step_store = {}
                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)
                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                current_loss = loss.detach().item()
                if epoch == 0:
                    loss_list.append(current_loss)
                else:
                    loss_total -= loss_list[step]
                    loss_list[step] = current_loss
                loss_total += current_loss
                avr_loss = loss_total / len(loss_list)
                logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                # detach attention_losses dict
                attention_losses = {k: v.detach().item() for k, v in attention_losses.items()}

                progress_bar.set_postfix(**logs)
                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})
                if args.logging_dir is not None:
                    logs = self.generate_step_logs(args, current_loss, avr_loss, lr_scheduler, keys_scaled, mean_norm, maximum_norm, **attention_losses)
                    accelerator.log(logs, step=global_step)
                    if is_main_process:
                        #wandb_tracker = accelerator.get_tracker("wandb")
                        wandb.log(logs)
                if global_step >= args.max_train_steps:
                    break
            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_total / len(loss_list)}
                accelerator.log(logs, step=epoch + 1)
            accelerator.wait_for_everyone()
            # 指定エポックごとにモデルを保存
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)
                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)
                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
            #self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)
            efficient_layers = args.efficient_layer.split(",")


            #unwrapped_nw = accelerator.unwrap_model(network)
            #weights_sd = unwrapped_nw.state_dict()
            if is_main_process:
                weights_sd = network.state_dict()
                layer_names = weights_sd.keys()
                for layer_name in layer_names:
                    score = 0
                    for efficient_layer in efficient_layers:
                        if efficient_layer in layer_name:
                            score += 1
                    if score == 0:
                        weights_sd[layer_name] = weights_sd[layer_name] * 0
                    weights_sd[layer_name] = weights_sd[layer_name].cpu()
                import copy
                vae_copy = copy.deepcopy(vae_org)
                text_encoder_copy = copy.deepcopy(text_encoder_org)
                unet_copy = copy.deepcopy(unet_org)
                temp_network, weights_sd = network_module.create_network_from_weights(multiplier=1,
                                                                                      file=None,
                                                                                      block_wise=[1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                                                     1, 1, 1, 1, 1, 1, 1, 1],
                                                                                      vae=vae_copy,
                                                                                      text_encoder=text_encoder_copy,
                                                                                      unet=unet_copy,
                                                                                      weights_sd=weights_sd,
                                                                                      for_inference=False)
                # load state dict
                print(f'right before loadint state, weights_sd is {weights_sd}')
                temp_network.load_state_dict(weights_sd, False)
                temp_network.to(weight_dtype)
                self.sample_images(accelerator, args, epoch + 1, global_step,
                                   accelerator.device, vae_copy, tokenizer,
                                   text_encoder_copy,
                                   unet_copy,
                                   efficient=True)

                print(f"temporary network are loaded")





            if attention_storer is not None:
                attention_storer.step_store = {}
            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()

        if is_main_process and args.save_state:
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)
            print("model saved.")

            # saving attn loss
            import csv
            attn_loss_save_dir = os.path.join(args.output_dir, 'record', f'{args.wandb_run_name}_attn_loss.csv')
            with open(attn_loss_save_dir, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(attn_loss_records)

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
    parser.add_argument("--class_caption_dir", type=str, default='./sentence_datas/cat_sentence_100.txt' )
    # class_caption
    parser.add_argument("--class_caption", type=str, default='girl')
    parser.add_argument("--heatmap_loss", action='store_true')
    parser.add_argument("--attn_loss_ratio", type=float, default=1.0)
    parser.add_argument("--mask_dir", type=str)
    parser.add_argument("--pretraining_epochs", type=int, default = 10)
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
    parser.add_argument('--class_token', default='cat', type=str)
    parser.add_argument('--unet_net_key_names', default='proj_in,ff_net', type=str)
    parser.add_argument("--te_freeze", action='store_true')
    parser.add_argument("--efficient_layer", type=str)
    args = parser.parse_args()
    # overwrite args.attn_loss_layers if only_second_training, only_third_training, second_third_training, first_second_third_training is True
    if args.only_second_training:
        args.attn_loss_layers = 'down_blocks_2,up_blocks_1'
    elif args.only_third_training:
        args.attn_loss_layers = 'down_blocks_1,up_blocks_2'
    elif args.first_second_training:
        args.attn_loss_layers = 'mid,down_blocks_2,up_blocks_1'
    elif args.second_third_training:
        args.attn_loss_layers = 'down_blocks_2,up_blocks_1,down_blocks_1,up_blocks_2'
    elif args.first_second_third_training:
        args.attn_loss_layers = 'mid,down_blocks_2,up_blocks_1,down_blocks_1,up_blocks_2'
    else :
        args.attn_loss_layers = 'all'

    # if any of only_second_training, only_third_training, second_third_training, first_second_third_training is True, print message to notify user that args.attn_loss_layers is overwritten
    if args.only_second_training or args.only_third_training or args.second_third_training or args.first_second_third_training:
        print(f"args.attn_loss_layers is overwritten to {args.attn_loss_layers} because only_second_training, only_third_training, second_third_training, first_second_third_training is True")

    if args.wandb_init_name is not None:
        tempfile_new = tempfile.NamedTemporaryFile()
        print(f"Created temporary file: {tempfile_new.name}")
        if args.wandb_log_template_path is not None:
            with open(args.wandb_log_template_path, 'r', encoding='utf-8') as f:
                lines = f.read()
        else:
            lines = '''[wandb]
    name = "{0}"'''
        tempfile_path = tempfile_new.name
        with open(tempfile_path, 'w', encoding='utf-8') as f:
            # format
            f.write(lines.format(args.wandb_init_name))
        args.log_tracker_config = tempfile_path #overwrite
    args = train_util.read_config_from_file(args, parser)
    trainer = NetworkTrainer()
    trainer.train(args)

