accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_0_1_config --main_process_port 20164 train_network_pretraining.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --wandb_init_name iom_pretraining --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
                  --wandb_run_name text_pretraining_unet_down_2_mid_up_1_low_repeat_test_heatmap_loss\
                  --seed 42 --log_with wandb --output_dir ./result/perfusion_experiment/cat/text_pretraining_unet_down_2_mid_up_1_low_repeat_test_heatmap_loss \
                  --max_token_length 225 --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
                  --train_data_dir /data7/sooyeon/MyData/perfusion_dataset/iom \
                  --mask_dir /data7/sooyeon/MyData/perfusion_dataset/iom_mask \
                  --class_caption 'cat' \
                  --class_caption_dir ./sentence_datas/cat_sentence_100.txt \
                  --class_token cat --trg_concept iom \
                  --network_module networks.lora --resolution 512,512 --net_key_names text \
                  --network_dim 64 --network_alpha 4 \
                  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
                  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
                  --pretraining_epochs 10 --unet_net_key_names 'down_blocks_2,mid,up_blocks_1' \
                  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/LyCORIS/test/test_iom.txt \
                  --heatmap_loss --mask_threshold 0.5 --heatmap_backprop --attn_loss_layers 'all'







accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_2_3_config --main_process_port 22364 train_network_pretraining.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --wandb_init_name iom_pretraining --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
                  --wandb_run_name text_pretraining_unet_down_2_mid_up_1_low_repeat \
                  --seed 42 --log_with wandb --output_dir ./result/perfusion_experiment/cat/text_pretraining_unet_down_2_mid_up_1_low_repeat \
                  --max_token_length 225 --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
                  --train_data_dir /data7/sooyeon/MyData/perfusion_dataset/iom \
                  --mask_dir /data7/sooyeon/MyData/perfusion_dataset/iom_mask \
                  --class_caption 'cat' \
                  --class_caption_dir ./sentence_datas/cat_sentence_100.txt \
                  --class_token cat --trg_concept iom \
                  --network_module networks.lora --resolution 512,512 --net_key_names text \
                  --network_dim 64 --network_alpha 4 \
                  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
                  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
                  --pretraining_epochs 10 --unet_net_key_names 'down_blocks_2,mid,up_blocks_1' \
                  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/LyCORIS/test/test_iom.txt \
                  --heatmap_loss --mask_threshold 0.5 --attn_loss_layers 'all'