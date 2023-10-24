CUDA_VISIBLE_DEVICES=6 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/iom_experiment/two_image_repeat_ten_max/iom_base_test/iom-000050.safetensors' \
      --prompt 'iom, sitting in chair' \
      --outdir ./result/iom_experiment/two_image_repeat_ten_max/iom_base_test/inference_attention/iom_epoch_50 --seed 42 --trg_token 'iom' \
      --negative_prompt 'drawn by bad-artist, sketch by bad-artist-anime, ugly, worst quality, poor details,bad-hands'

CUDA_VISIBLE_DEVICES=0 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/iom_experiment/one_image/iom_detailed_second/iom-000053.safetensors' \
      --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/iom_inference.txt \
      --outdir attn_test/20231024_result/iom/iom_base_epoch_53 --trg_token 'iom'