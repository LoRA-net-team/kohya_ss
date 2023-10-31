CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/perfusion_experiment/cat/text_pretraining_unet_down_2_mid_up_1_low_repeat/epoch-000002.safetensors' \
      --outdir './result/perfusion_experiment/cat/text_pretraining_unet_down_2_mid_up_1_low_repeat/inference/epoch_2/' \
      --from_file '/data7/sooyeon/LyCORIS/test/iom_inference.txt'