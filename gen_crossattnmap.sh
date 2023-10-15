CUDA_VISIBLE_DEVICES=0 python gen_camap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/one_image/haibara_3_1_image_base/haibara-000026.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir 20231011_result/haibara_experience/haibara_3_1_image_base/haibara_epoch_26 --seed 42 --trg_token 'haibara' \
      --negative_prompt 'worst quality, mutated hand, blurry'

#---------------
CUDA_VISIBLE_DEVICES=0 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/haibara_experience/one_image/haibara_3_1_image_base/haibara-000026.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir 20231011_result/haibara_experience/haibara_3_1_image_base/haibara_epoch_26 --seed 42 --trg_token 'haibara'