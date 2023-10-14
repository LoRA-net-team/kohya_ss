CUDA_VISIBLE_DEVICES=0 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_1_image_base/haibara-000033.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231014_result/haibara_3_1_image_base/haibara_epoch_33 --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_1_image_mask_10_mean/haibara-000040.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231014_result/haibara_3_1_image_mask_10_mean/haibara_epoch_40 --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_4_image_base/haibara-000045.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231014_result/haibara_3_4_image_base/haibara_epoch_45 --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=3 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_4_image_mask_10_mean/haibara-000034.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231014_result/haibara_3_4_image_mask_10_mean/haibara_epoch_34 --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=3 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_19_image_base/haibara-000015.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231014_result/haibara_3_19_image_base/haibara_epoch_15 --seed 42 --trg_token 'haibara'
# ------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=5 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/haibara_3_19_image_mask_10_mean/haibara-000020.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir attn_test/20231014_result/haibara_3_19_image_mask_10_mean/haibara_epoch_20 --seed 42 --trg_token 'haibara'