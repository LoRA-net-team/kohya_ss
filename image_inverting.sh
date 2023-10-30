python image_inverting.py --device cuda:5  --process_title parksooyeon \
                          --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
                          --concept_image_folder /data7/sooyeon/MyData/perfusion_dataset/cat \
                          --prompt 'cat wearing like a wizard' \
                          --negative_prompt 'low quality, worst quality, bad anatomy,bad composition, poor, low effort' \
                          --output_dir ./result/inference_result/perfusion_experiment/cat/random_init_guidance_8_ddim_50_sef_mean --seed 42 --guidance_scale 8 \
                          --num_ddim_steps 50 --min_value 12 --max_self_input_time 10 --self_key_control

python image_inverting.py --device cuda:4  --process_title parksooyeon \
                          --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
                          --concept_image /data7/sooyeon/MyData/perfusion_dataset/cat/1_no_background.jpg \
                          --prompt 'cat wearing like a wizard' \
                          --negative_prompt 'low quality, worst quality, bad anatomy,bad composition, poor, low effort' \
                          --output_dir ./result/inference_result/perfusion_experiment/cat/random_init_guidance_8_ddim_50_keys_values_without_background_img --seed 42 --guidance_scale 8 \
                          --num_ddim_steps 50 --min_value 12 --max_self_input_time 10 --self_key_control


