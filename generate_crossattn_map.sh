CUDA_VISIBLE_DEVICES=2 python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_3_blur_mask_output/jungwoo-000015.safetensors" \
      --prompt "jw, white_background, standing" --outdir 20231008_result/jungwoo_3_blur_mask_output/jw --seed 42 --trg_token 'jw'










# ---------------------------------------------------------------------------------------------------------------------------------------
python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights "./result/jungwoo_experience/jungwoo_1_attnloss_10/jungwoo-000021.safetensors" \
      --prompt "jungwoo, white_background" --outdir 20231007_result/jungwoo_1_attnloss_10 --seed 42 --trg_token 'jungwoo'

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