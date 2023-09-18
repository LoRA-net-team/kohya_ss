python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights /data7/sooyeon/pretrained_lora/AkaneV1.2.safetensors \
      --prompt "Akane, white_background" --outdir 20230930_result --seed 42


      --negative_prompt "easynegative, badhandv4,low quality, lowres:1.4, worst quality, blurry, blurry background," \