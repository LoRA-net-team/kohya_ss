python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights /data7/sooyeon/pretrained_lora/AkaneV1.2.safetensors \
      --seed 42