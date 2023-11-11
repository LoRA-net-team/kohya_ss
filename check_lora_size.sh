accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_config/gpu_4_5_config --main_process_port 24564 only_inference.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --max_token_length 225 \
                  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
                  --output_dir ../result/perfusion_experiment/cat/iom_pretrain_sen_10_unet_heatmap_backprop_infer_inlayers \
                  --save_folder_name 'test' \
                  --network_module networks.lora --efficient_layer 'text,unet'


                   \
                  --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/MyData/perfusion_dataset/cat/test_other.txt

                  vi /data7/sooyeon/MyData/perfusion_dataset/cat/test_other.txt