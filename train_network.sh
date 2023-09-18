accelerate launch --config_file gpu_7_0_config --main_process_port 26578 train_network.py \
            --pretrained_model_name_or_path pretrained/animefull-final-pruned-fp16.safetensors \
            --shuffle_caption --caption_extension ".txt" \
            --random_crop --resolution "768,768" --enable_bucket --bucket_no_upscale \
            --save_every_n_epochs 1 --train_batch_size 4 --max_token_length 225 --xformers \
            --max_train_steps 2880 \
            --persistent_data_loader_workers \
            --seed 42 --sample_every_n_epochs 1 \
            --gradient_checkpointing \
            --network_module networks.lora \
            --train_data_dir /data7/sooyeon/MyData/agnes_tachyon_small \
            --lr_warmup_steps 144 --unet_lr 0.0003 --text_encoder_lr 0.00015 --network_dim 64 --network_alpha 64.0 \
            --output_dir ./result/model_agnes_tachyon --output_name agnes_tachyon \
            --logging_dir ./result/logs \
            --noise_offset 0.0357 --optimizer_type AdamW --learning_rate 0.0003 \
            --lr_scheduler cosine_with_restarts \
            --wandb_init_name compare_test --run_name agnes_tachyon_small \
            --sample_prompts test/test_agnes_tachyon.txt

# ------------------------------------------
accelerate launch --config_file gpu_7_0_config --main_process_port 26578 train_network_reg.py \
            --pretrained_model_name_or_path pretrained/animefull-final-pruned-fp16.safetensors \
            --shuffle_caption --caption_extension ".txt" \
            --random_crop --resolution "768,768" --enable_bucket --bucket_no_upscale \
            --save_every_n_epochs 1 --train_batch_size 4 --max_token_length 225 --xformers \
            --max_train_steps 2880 \
            --persistent_data_loader_workers \
            --seed 42 --sample_every_n_epochs 1 \
            --gradient_checkpointing \
            --network_module networks.lora2 \
            --train_data_dir /data7/sooyeon/MyData/agnes_tachyon_small \
            --lr_warmup_steps 144 --unet_lr 0.0003 --text_encoder_lr 0.00015 --network_dim 64 --network_alpha 64.0 \
            --output_dir ./result/model_agnes_tachyon --output_name agnes_tachyon \
            --logging_dir ./result/logs \
            --noise_offset 0.0357 --optimizer_type AdamW --learning_rate 0.0003 \
            --lr_scheduler cosine_with_restarts \
            --wandb_init_name compare_test --run_name agnes_tachyon_small \
            --sample_prompts test/test_agnes_tachyon.txt

# ------------------------------------------