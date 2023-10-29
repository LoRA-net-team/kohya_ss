python image_inverting.py --device cuda:4  --process_title parksooyeon \
                          --concept_image /data7/sooyeon/MyData/perfusion_dataset/cat/1.jpg \
                          --prompt 'cat, wearing like a chef' \
                          --negtive_prompt 'low quality, worst quality, bad anatomy,bad composition, poor, low effort'
                          --output_dir ./result/perfusion_experiment/cat/20231029 \
                          --max_self_input_time 10 --num_ddim_steps 30 --seed 42
