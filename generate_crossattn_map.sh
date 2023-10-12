CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/jungwoo_experience/jungwoo_3_attn_loss_layerwise_detail/jungwoo-000009.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo_inference.txt' \
      --outdir 20231011_result/jungwoo_3_attn_loss_layerwise_detail/jw --seed 42 --trg_token 'jw'

# ---------------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=3 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/jungwoo_experience/jungwoo_3_attn_loss_test_2/jungwoo-000016.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo_inference.txt' \
      --outdir 20231011_result/jungwoo_3_attn_loss_test_2/jw --seed 42 --trg_token 'jw'
# ---------------------------------------------------------------------------------------------------------------------------------------




CUDA_VISIBLE_DEVICES=3 python generate_crossattn_map.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \


      --prompt "jw, male_focused, center, smiling, wearing_shirt" --outdir 20231011_result/jungwoo_3_blur_mask_10_mean/jw --seed 42 --trg_token 'jw'
