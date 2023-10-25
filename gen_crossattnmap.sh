CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/iom_experiment/one_image/iom_test_task_loss_high_repeat/iom-000011.safetensors' \
      --outdir ./result/20231025/inference_attention/iom_epoch_11_except_unet --seed 42 --trg_token 'iom' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/iom_inference.txt' \
      --exception_layer 'unet'



      --prompt 'iom, sitting in chair' \




      --negative_prompt 'drawn by bad-artist, sketch by bad-artist-anime, ugly, worst quality, poor details,bad-hands'

CUDA_VISIBLE_DEVICES=0 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/iom_experiment/one_image/iom_detailed_second/iom-000053.safetensors' \
      --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/iom_inference.txt \
      --outdir attn_test/20231024_result/iom/iom_base_epoch_53 --trg_token 'iom'

