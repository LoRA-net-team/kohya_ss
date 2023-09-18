accelerate launch --config_file gpu_3_4_config --main_process_port 23478 train_network_reg.py \
            --pretrained_model_name_or_path pretrained/animefull-final-pruned-fp16.safetensors \
            --shuffle_caption --caption_extension ".txt" \
            --random_crop --resolution "768,768" --enable_bucket --bucket_no_upscale \
            --save_every_n_epochs 1 --train_batch_size 4 --max_token_length 225 --xformers \
            --max_train_steps 2880 \
            --persistent_data_loader_workers \
            --seed 42 --sample_every_n_epochs 1 \
            --gradient_checkpointing \
            --network_module networks.lora2 \
            --train_data_dir /data7/sooyeon/MyData/haibara \
            --lr_warmup_steps 144 --unet_lr 0.0003 --text_encoder_lr 0.00015 --network_dim 64 --network_alpha 64.0 \
            --logging_dir ./result/logs \
            --noise_offset 0.0357 --optimizer_type AdamW --learning_rate 0.0003 \
            --lr_scheduler cosine_with_restarts \
            --reg_loss_weight 1.0 \
            --output_dir ./result/haibara_reg_test_reg_1.0_with_conv --output_name haibara \
            --wandb_init_name haibara_reg_test --run_name reg_1.0_with_conv \
            --sample_prompts test/test_haibara.txt


accelerate launch --config_file gpu_7_0_config --main_process_port 25778 train_db.py \
            --pretrained_model_name_or_path pretrained/animefull-final-pruned-fp16.safetensors \
            --shuffle_caption --caption_extension ".txt" \
            --random_crop --resolution "768,768" --enable_bucket --bucket_no_upscale \
            --save_every_n_epochs 1 --train_batch_size 4 --max_token_length 225 --xformers \
            --max_train_steps 2880 \
            --persistent_data_loader_workers \
            --seed 42 --sample_every_n_epochs 1 \
            --gradient_checkpointing \
            --network_module networks.lora2 \
            --train_data_dir /data7/sooyeon/MyData/haibara \
            --lr_warmup_steps 144 --unet_lr 0.00003 --text_encoder_lr 0.000015 --network_dim 64 --network_alpha 64.0 \
            --logging_dir ./result/logs \
            --noise_offset 0.0357 --optimizer_type AdamW --learning_rate 0.0003 \
            --lr_scheduler cosine_with_restarts \
            --reg_loss_weight 0.6 \
            --output_dir ./result/haibara_reg_test_reg_0.6_with_conv_low_lr --output_name haibara \
            --wandb_init_name haibara_reg_test --run_name reg_0.6_with_conv_low_lr \
            --sample_prompts test/test_haibara.txt



























accelerate launch --config_file gpu_3_4_config --main_process_port 25678 train_network_reg.py \
            --pretrained_model_name_or_path pretrained/animefull-final-pruned-fp16.safetensors \
            --shuffle_caption --caption_extension ".txt" \
            --random_crop --resolution "768,768" --enable_bucket --bucket_no_upscale \
            --save_every_n_epochs 1 --train_batch_size 4 --max_token_length 225 --xformers \
            --max_train_steps 2880 \
            --persistent_data_loader_workers \
            --seed 42 --sample_every_n_epochs 1 \
            --gradient_checkpointing \
            --network_module networks.lora2 \
            --train_data_dir /data7/sooyeon/MyData/haibara \
            --lr_warmup_steps 144 --unet_lr 0.0003 --text_encoder_lr 0.00015 --network_dim 64 --network_alpha 64.0 \
            --logging_dir ./result/logs \
            --noise_offset 0.0357 --optimizer_type AdamW --learning_rate 0.0003 \
            --lr_scheduler cosine_with_restarts \
            --reg_loss_weight 0.2 \
            --output_dir ./result/haibara_reg_test_reg_0.2 --output_name haibara \
            --wandb_init_name haibara_reg_test --run_name reg_0.2 \
            --sample_prompts test/test_haibara.txt


accelerate launch --config_file gpu_5_6_config --main_process_port 23578 train_network_reg.py \
            --pretrained_model_name_or_path pretrained/animefull-final-pruned-fp16.safetensors \
            --shuffle_caption --caption_extension ".txt" \
            --random_crop --resolution "768,768" --enable_bucket --bucket_no_upscale \
            --save_every_n_epochs 1 --train_batch_size 4 --max_token_length 225 --xformers \
            --max_train_steps 2880 \
            --persistent_data_loader_workers \
            --seed 42 --sample_every_n_epochs 1 \
            --gradient_checkpointing \
            --network_module networks.lora2 \
            --train_data_dir /data7/sooyeon/MyData/haibara \
            --lr_warmup_steps 144 --unet_lr 0.0003 --text_encoder_lr 0.00015 --network_dim 64 --network_alpha 64.0 \
            --logging_dir ./result/logs \
            --noise_offset 0.0357 --optimizer_type AdamW --learning_rate 0.0003 \
            --lr_scheduler cosine_with_restarts \
            --reg_loss_weight 0.4 \
            --output_dir ./result/haibara_reg_test_reg_0.4 --output_name haibara \
            --wandb_init_name haibara_reg_test --run_name reg_0.4 \
            --sample_prompts test/test_haibara.txt
accelerate launch --config_file gpu_7_0_config --main_process_port 24578 train_network_reg.py \
            --pretrained_model_name_or_path pretrained/animefull-final-pruned-fp16.safetensors \
            --shuffle_caption --caption_extension ".txt" \
            --random_crop --resolution "768,768" --enable_bucket --bucket_no_upscale \
            --save_every_n_epochs 1 --train_batch_size 4 --max_token_length 225 --xformers \
            --max_train_steps 2880 \
            --persistent_data_loader_workers \
            --seed 42 --sample_every_n_epochs 1 \
            --gradient_checkpointing \
            --network_module networks.lora2 \
            --train_data_dir /data7/sooyeon/MyData/haibara \
            --lr_warmup_steps 144 --unet_lr 0.0003 --text_encoder_lr 0.00015 --network_dim 64 --network_alpha 64.0 \
            --output_dir ./result/model_haibara_layer_test --output_name haibara \
            --logging_dir ./result/logs \
            --noise_offset 0.0357 --optimizer_type AdamW --learning_rate 0.0003 \
            --lr_scheduler cosine_with_restarts \
            --reg_loss_weight 0.6 \
            --output_dir ./result/haibara_reg_test_reg_0.2 --output_name haibara \
            --wandb_init_name haibara_reg_test --run_name reg_0.6 \
            --sample_prompts test/test_haibara.txt