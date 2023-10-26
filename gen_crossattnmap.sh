# proj_in 와 ff_net 레이어를 제외하고 학습해보기
# 즉, proj_in과 ff_net 는 특정 text 만 관여가 되면 특정 포즈가 되도록 하는 일만 하고 semantic 을 학습하지 못한다.

CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/iom_experiment/one_image/iom_test_task_loss_high_repeat/iom-000011.safetensors' \
      --outdir ./result/20231025/inference_attention/iom_epoch_11_with_te_proj_in_ff_net --seed 42 --trg_token 'iom' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/iom_inference.txt' \
      --exception_layer 'text,proj_in,ff_net'



      --prompt 'iom, sitting in chair' \




      --negative_prompt 'drawn by bad-artist, sketch by bad-artist-anime, ugly, worst quality, poor details,bad-hands'

CUDA_VISIBLE_DEVICES=0 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/iom_experiment/one_image/iom_detailed_second/iom-000053.safetensors' \
      --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/iom_inference.txt \
      --outdir attn_test/20231024_result/iom/iom_base_epoch_53 --trg_token 'iom'

