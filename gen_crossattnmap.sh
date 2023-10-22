CUDA_VISIBLE_DEVICES=6 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/iom_experiment/two_image_repeat_ten_max/iom_base_test/iom-000050.safetensors' \
      --prompt 'iom, sitting in chair' \
      --outdir ./result/iom_experiment/two_image_repeat_ten_max/iom_base_test/inference_attention/iom_epoch_50 --seed 42 --trg_token 'iom' \
      --negative_prompt 'drawn by bad-artist, sketch by bad-artist-anime, ugly, worst quality, poor details,bad-hands'