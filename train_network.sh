accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_6_7_config --main_process_port 26764 train_network_pretraining.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --seed 42 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
                  --max_token_length 225 \
                  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
                  --wandb_init_name td_pretraining \
                  --wandb_run_name te_pretrain_581_sen_25_preserving_class_token_pretraining_freeze_crossattn_kv \
                  --output_dir ./result/perfusion_experiment/teddy_bear/te_pretrain_581_sen_15_preserving_class_token_pretraining_freeze_crossattn_kv \
                  --train_data_dir /data7/sooyeon/MyData/perfusion_dataset/teddy_one \
                  --mask_dir /data7/sooyeon/MyData/perfusion_dataset/teddy_one_mask \
                  --class_token 'teddy bear' --class_caption 'teddy bear' --trg_concept td \
                  --class_caption_dir ./sentence_datas/teddy_bear_sentence_581.txt \
                  --network_module networks.lora --resolution 512,512 --net_key_names text \
                  --network_dim 64 --network_alpha 4 \
                  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
                  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
                  --pretraining_epochs 1 --unet_net_key_names 'proj_in,ff_net,attn1,attn2_to_q,attn2_to_out' \
                  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/LyCORIS/test/test_td.txt \
                  --heatmap_loss --mask_threshold 0.5 --attn_loss_layers 'all' --efficient_layer 'text,proj_in,ff_net'

























accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_4_5_config --main_process_port 24564 train_network_pretraining.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --seed 42 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
                  --max_token_length 225 \
                  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
                  --wandb_init_name td_pretraining \
                  --wandb_run_name te_pretrain_581_sen_15_preserving_class_token_pretraining_freeze_unet \
                  --output_dir ./result/perfusion_experiment/teddy_bear/te_pretrain_581_sen_15_preserving_class_token_pretraining_freeze_unet \
                  --train_data_dir /data7/sooyeon/MyData/perfusion_dataset/teddy_one \
                  --mask_dir /data7/sooyeon/MyData/perfusion_dataset/teddy_one_mask \
                  --class_token 'teddy bear' --class_caption 'teddy bear' --trg_concept td \
                  --class_caption_dir ./sentence_datas/teddy_bear_sentence_581.txt \
                  --network_module networks.lora --resolution 512,512 --net_key_names text \
                  --network_dim 64 --network_alpha 4 \
                  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
                  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
                  --pretraining_epochs 15 --unet_net_key_names 'unet' \
                  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/LyCORIS/test/test_td.txt \
                  --heatmap_loss --mask_threshold 0.5 --attn_loss_layers 'all' --te_freeze