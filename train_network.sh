accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_0_1_config --main_process_port 20164 train_network_pretraining.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --seed 42 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
                  --max_token_length 225 \
                  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
                  --wandb_init_name td_pretraining \
                  --wandb_run_name te_pretrain_581_sen_10_crossatten_unefficient \
                  --output_dir ./result/perfusion_experiment/teddy_bear/te_pretrain_581_sen_10_crossatten_unefficient \
                  --train_data_dir /data7/sooyeon/MyData/perfusion_dataset/teddy_one \
                  --mask_dir /data7/sooyeon/MyData/perfusion_dataset/teddy_one_mask \
                  --class_token 'teddy bear' --class_caption 'teddy bear' --trg_concept td \
                  --class_caption_dir ./sentence_datas/teddy_bear_sentence_581.txt \
                  --network_module networks.lora --resolution 512,512 --net_key_names text \
                  --network_dim 64 --network_alpha 4 \
                  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
                  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
                  --pretraining_epochs 10 --unet_net_key_names 'unet' \
                  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/LyCORIS/test/test_td.txt \
                  --heatmap_loss --mask_threshold 0.5 --attn_loss_layers 'all' --unefficient_layer 'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k,down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v,down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k,down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v,mid_block_attentions_0_transformer_blocks_0_attn2_to_k,mid_block_attentions_0_transformer_blocks_0_attn2_to_v,up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k,up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v,up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k,up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v,up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_k,up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_v'

#
accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_0_1_config --main_process_port 20164 only_inference.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --max_token_length 225 \
                  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
                  --output_dir ./result/perfusion_experiment/teddy_bear/te_pretrain_581_sen_10_attn1_to_k_attn1_to_v_attn2_to_k_attn2_to_v \
                  --network_module networks.lora --efficient_layer 'text,down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k,down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v,down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k,down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v,down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k,down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v,down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k,down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v,mid_block_attentions_0_transformer_blocks_0_attn1_to_k,mid_block_attentions_0_transformer_blocks_0_attn1_to_v,mid_block_attentions_0_transformer_blocks_0_attn2_to_k,mid_block_attentions_0_transformer_blocks_0_attn2_to_v,up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k,up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v,up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k,up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v,up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k,up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v,up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k,up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v,up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_k,up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_v,up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_k,up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_v'