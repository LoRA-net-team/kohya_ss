accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_2_3_config --main_process_port 22364 sy.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --max_token_length 225 \
                  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
                  --output_dir ./result/perfusion_experiment/teddy_bear/te_pretrain_581_sen_10_unet_inlayers_cond_inference_classcaption_preserving_on_inlayer_heatmap_backprop_attn_loss_ratio_10_highrepeat \
                  --save_folder_name 'inference_te_inblocks_text_conditional_zero' \
                  --network_module networks.lora --efficient_layer 'text,down_blocks_2,mid,up_blocks_1' \
                  --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/LyCORIS/test/test_td.txt