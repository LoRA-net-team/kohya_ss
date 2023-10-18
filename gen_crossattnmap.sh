CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/one_image/name_9/haibara_9_continuous_binary_mask_only_second_training_10/haibara-000044.safetensors' \
      --prompt 'haibara,masterpiece, best quality, 1girl, bangs, blue_eyes, cowboy_shot, looking_at_viewer, school uniform, solo, highly detailed, solo, cowboy shot, palm tree, waves, smile, sitting on a chair' \
      --outdir ./result/haibara_experience/one_image/name_9/haibara_9_continuous_binary_mask_only_second_training_10/inference_attention/haibara_epoch_44_one_prompt --seed 42 --trg_token 'haibara' \
      --negative_prompt 'drawn by bad-artist, sketch by bad-artist-anime, ugly, worst quality, poor details,bad-hands'
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/one_image/name_4/haibara_4_continuous_binary_mask_only_second_training_10/haibara-000044.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir ./result/haibara_experience/one_image/name_4/haibara_4_c ontinuous_binary_mask_only_second_training_10/inference_attention/haibara_epoch_26 --seed 42 --trg_token 'haibara' \
      --negative_prompt 'worst quality, mutated hand, blurry'


CUDA_VISIBLE_DEVICES=1 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/one_image/name_3/haibara_3_1_image_continuous_binary_mask/haibara-000035.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir ./result/haibara_experience/one_image/name_3/haibara_3_1_image_continuous_binary_mask/inference_attention/haibara_epoch_35 --seed 42 --trg_token 'haibara' \
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