python image_inverting.py --device cuda:4  --process_title parksooyeon \
                          --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
                          --concept_image /data7/sooyeon/MyData/perfusion_dataset/cat/1.jpg \
                          --prompt 'cat, wearing like a chef' \
                          --negative_prompt 'low quality, worst quality, bad anatomy,bad composition, poor, low effort' \
                          --output_dir ./result/perfusion_experiment/cat/20231029 \
                          --max_self_input_time 10 --num_ddim_steps 50 --seed 42 --guidance_scale 7.5