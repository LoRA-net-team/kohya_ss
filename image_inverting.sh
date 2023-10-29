python image_inverting.py --device cuda:5  --process_title parksooyeon \
                          --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
                          --concept_image /data7/sooyeon/MyData/perfusion_dataset/cat/1.jpg \
                          --prompt 'cat, wearing like a chef' \
                          --negative_prompt 'low quality, worst quality, bad anatomy,bad composition, poor, low effort' \
                          --output_dir ./result/perfusion_experiment/cat/20231029_random_init_guidance_8_min_3_max_30_only_value_control --seed 42 --guidance_scale 8 \
                          --num_ddim_steps 50 --min_value 3 --max_self_input_time 30



                           --max_self_input_time 900  --min_value 10
                           self_input_time < max_self_input_time and self_input_time > args.