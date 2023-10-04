python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_base_1/jungwoo-000020.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231002_result/jungwoo_base_1 --seed 42 --trg_token 'jungwoo'
# ---------------------------------------------------------------------------------------------------------------------------------------
python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_base_2/jungwoo-000024.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231002_result/jungwoo_base_2 --seed 42 --trg_token 'jungwoo'
# ---------------------------------------------------------------------------------------------------------------------------------------
python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_base_3/jungwoo-000018.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231002_result/jungwoo_base_3 --seed 42 --trg_token 'jungwoo'
# ---------------------------------------------------------------------------------------------------------------------------------------
python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_base_4/jungwoo-000017.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231002_result/jungwoo_base_4 --seed 42 --trg_token 'jungwoo'










# ---------------------------------------------------------------------------------------------------------------------------------------
python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_1_attntion_loss_per_block/jungwoo-000025.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231002_result/jungwoo_1_attntion_loss_per_block --seed 42 --trg_token 'jungwoo'
# ---------------------------------------------------------------------------------------------------------------------------------------
python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_2_attntion_loss_per_block/jungwoo-000018.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231002_result/jungwoo_2_attntion_loss_per_block --seed 42 --trg_token 'jungwoo'
# ---------------------------------------------------------------------------------------------------------------------------------------
python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_3_attntion_loss_per_block/jungwoo-000026.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231002_result/jungwoo_3_attntion_loss_per_block --seed 42 --trg_token 'jungwoo'
# ---------------------------------------------------------------------------------------------------------------------------------------
python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_4_attntion_loss_per_block/jungwoo-000017.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231002_result/jungwoo_4_attntion_loss_per_block --seed 42 --trg_token 'jungwoo'