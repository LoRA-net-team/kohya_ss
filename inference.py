import importlib, wandb
import argparse
import gc
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
import toml

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import model_util
from library import train_util
from library.train_util import (
    DreamBoothDataset,
)
from library import config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library import huggingface_util
from library import custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
)

def main(args):

    print(f' step 1. session id & seed')
    session_id = random.randint(0, 2**32)
    training_started_at = time.time()
    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    print(f' step 2. tokenizer')
    tokenizer = train_util.load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

    print(f' step 3. preparing accelerator')
    accelerator = train_util.prepare_accelerator(args)

    print(f' step 4. mixed precision')
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
    model_version = model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization)
    # text_encoder is List[CLIPTextModel] or CLIPTextModel
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    # モデルに xformers とか memory efficient attention を組み込む
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
    if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
        vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # 差分追加学習のためにモデルを読み込む
    sys.path.append(os.path.dirname(__file__))
    accelerator.print("import network module:", args.network_module)
    network_module = importlib.import_module(args.network_module)

    if args.base_weights is not None:
        # base_weights が指定されている場合は、指定された重みを読み込みマージする
        for i, weight_path in enumerate(args.base_weights):
            if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                multiplier = 1.0
            else:
                multiplier = args.base_weights_multiplier[i]
            accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")
            module, weights_sd = network_module.create_network_from_weights(multiplier, weight_path, vae, text_encoder, unet, for_inference=True)
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

    for t_enc in text_encoders:
        t_enc.to(accelerator.device)

    # prepare network
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value

    # if a new network is added in future, add if ~ then blocks for each network (;'∀')
    if args.dim_from_weights:
        network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
    else:
        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            vae,
            text_encoder,
            unet,
            neuron_dropout=args.network_dropout,
            **net_kwargs,)
    if network is None:
        return

    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
        print(
            "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
        )
        args.scale_weight_norms = False

    train_unet = not args.network_train_text_encoder_only
    train_text_encoder = not args.network_train_unet_only
    network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        accelerator.print(f"load network weights from {args.network_weights}: {info}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        for t_enc in text_encoders:
            t_enc.gradient_checkpointing_enable()
        del t_enc
        network.enable_gradient_checkpointing()  # may have no effect

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        network.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        network.to(weight_dtype)

    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)
    for t_enc in text_encoders:
        t_enc.requires_grad_(False)

    # acceleratorがなんかよろしくやってくれるらしい
    # TODO めちゃくちゃ冗長なのでコードを整理する
    if train_unet and train_text_encoder:
        if len(text_encoders) > 1:
            unet, t_enc1, t_enc2, network = accelerator.prepare(unet, text_encoders[0], text_encoders[1], network)
            text_encoder = text_encoders = [t_enc1, t_enc2]
            del t_enc1, t_enc2
        else:
            unet, text_encoder, network = accelerator.prepare(unet, text_encoder, network)
            text_encoders = [text_encoder]
    elif train_unet:
        unet, network = accelerator.prepare(unet, network)

    elif train_text_encoder:
        if len(text_encoders) > 1:
            t_enc1, t_enc2, network = accelerator.prepare(text_encoders[0], text_encoders[1])
            text_encoder = text_encoders = [t_enc1, t_enc2]
            del t_enc1, t_enc2
        else:
            text_encoder, network = accelerator.prepare(text_encoder, network)
            text_encoders = [text_encoder]

        unet.to(accelerator.device, dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
    else:
        network = accelerator.prepare(network)

    # transform DDP after prepare (train_network here only)
    text_encoders = train_util.transform_models_if_DDP(text_encoders)
    unet, network = train_util.transform_models_if_DDP([unet, network])

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

    train_util.sample_images(accelerator, args,
                             100,
                             0, accelerator.device, vae, tokenizer,
                             text_encoder, unet)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true",
                        help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む")
    parser.add_argument(
        "--v_parameterization", action="store_true",
        help="enable v-parameterization training / v-parameterization学習を有効にする"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )

    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument("--no_metadata", action="store_true",
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")

    parser.add_argument("--network_weights", type=str,
                        default="/data7/sooyeon/LyCORIS/LyCORIS/result/haibara_reg_test_reg_0.6/haibara-000019.safetensors")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument(
        "--network_dim", type=int, default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）"
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args", type=str, default=None, nargs="*",
        help="additional argmuments for network (key=value) / ネットワークへの追加の引数"
    )
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument(
        "--network_train_text_encoder_only", action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する"
    )
    parser.add_argument(
        "--training_comment", type=str, default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列"
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument("--wandb_init_name",type=str)
    parser.add_argument("--run_name",type=str)
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train(args)
