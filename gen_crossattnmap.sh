CUDA_VISIBLE_DEVICES=4 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/jungwoo_experience/jungwoo_3_base/jungwoo-000013.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo_inference.txt' \
      --outdir 20231011_result/jungwoo_3_base/jw --seed 42 --trg_token 'jw'

CUDA_VISIBLE_DEVICES=5 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_module networks.lora \
      --network_weights './result/jungwoo_experience/jungwoo_3_blur_mask_10_mean/jungwoo-000015.safetensors' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo_inference.txt' \
      --outdir 20231011_result/jungwoo_3_blur_mask_10_mean/jw --seed 42 --trg_token 'jw'