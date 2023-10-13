CUDA_VISIBLE_DEVICES=0 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_one_image_base/haibara-000033.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231013_result/haibara_3_one_image_base/haibara --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_one_image_mask/haibara-000059.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231013_result/haibara_3_one_image_mask/haibara --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_four_image_base/haibara-000045.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231013_result/haibara_3_four_image_base/haibara --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=3 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_four_image_mask/haibara-000037.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231013_result/haibara_3_four_image_mask/haibara --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------