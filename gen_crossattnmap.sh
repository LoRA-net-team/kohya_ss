# proj_in 와 ff_net 레이어를 제외하고 학습해보기
# 즉, proj_in과 ff_net 는 특정 text 만 관여가 되면 특정 포즈가 되도록 하는 일만 하고 semantic 을 학습하지 못한다.

CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/tdy_experiment/td_base_3/td.safetensors' \
      --outdir ./result/20231027/inference_attention/td_base_3_with_te_attn2_with_flowers --seed 42 --trg_token 'td' \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/td_inference.txt' \
      --exception_layer 'text,attn2'

vi /data7/sooyeon/LyCORIS/LyCORIS/test/td_inference.txt


cp /data7/sooyeon/LyCORIS/LyCORIS/test/iom_inference.txt /data7/sooyeon/LyCORIS/LyCORIS/test/td_inference.txttd-000005.safetensors




CUDA_VISIBLE_DEVICES=1 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_girl_sub_inference.txt' \
      --outdir attn_test/20231014_result/base_model/smile --seed 42 --trg_token 'haibara' \
      --network_module networks.lora \

      --from_file '/data7/sooyeon/LyCORIS/LyCORIS/test/test_haibara_inference.txt' \
      --outdir ./result/haibara_experience/one_image/name_3_without_caption/haibara_second_1/attn_inference/haibara_epoch_40



      --prompt 'iom, sitting in chair' \




      --negative_prompt 'drawn by bad-artist, sketch by bad-artist-anime, ugly, worst quality, poor details,bad-hands'

CUDA_VISIBLE_DEVICES=0 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/iom_experiment/one_image/iom_detailed_second/iom-000053.safetensors' \
      --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/iom_inference.txt \
      --outdir attn_test/20231024_result/iom/iom_base_epoch_53 --trg_token 'iom'

