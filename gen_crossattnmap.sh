CUDA_VISIBLE_DEVICES=0 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/one_image/name_8/haibara_3_1_8_image_base/haibara-000060.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir ./result/haibara_experience/one_image/name_8/haibara_3_1_8_image_base/inference_attention/haibara_epoch_60 --seed 42 --trg_token 'haibara' \
      --negative_prompt 'worst quality, mutated hand, blurry'
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/one_image/haibara_3_1_image_continuous_mask_1_preserve_1/haibara-000028.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir ./result/haibara_experience/one_image/haibara_3_1_image_continuous_mask_1_preserve_1/inference_attention/haibara_epoch_28 --seed 42 --trg_token 'haibara' \
      --negative_prompt 'worst quality, mutated hand, blurry'
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/four_image/haibara_3_4_image_continuous_mask/haibara-000142.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir ./result/haibara_experience/four_image/haibara_3_4_image_continuous_mask/inference_attention/haibara_epoch_142 --seed 42 --trg_token 'haibara' \
      --negative_prompt 'worst quality, mutated hand, blurry'
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=3 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/four_image/haibara_3_4_image_continuous_mask_preserving/haibara-000121.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir ./result/haibara_experience/four_image/haibara_3_4_image_continuous_mask_preserving/inference_attention/haibara_epoch_121 --seed 42 --trg_token 'haibara' \
      --negative_prompt 'worst quality, mutated hand, blurry'