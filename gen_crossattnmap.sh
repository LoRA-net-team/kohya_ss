CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/perfusion_experiment/teddy_bear/text_pretraining_unet_down_2_mid_up_1_low_repeat/epoch-000003.safetensors' \
      --outdir './result/perfusion_experiment/teddy_bear/text_pretraining_unet_down_2_mid_up_1_low_repeat/inference/epoch_3/' \
      --from_file '/data7/sooyeon/LyCORIS/test/td_inference.txt' \
      --efficient_layer 'text,down_blocks_2,mid,up_blocks_1'


      cp /data7/sooyeon/LyCORIS/test/iom_inference.txt /data7/sooyeon/LyCORIS/test/td_inference.txt
      vi