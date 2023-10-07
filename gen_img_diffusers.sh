cuda_visible_devices=2 python crossattnmap.py \
                    --network_module networks.lora \
                    --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
                    --network_weights "./result/jungwoo_experience/jungwoo_1_attnloss_10/jungwoo-000021.safetensors" \
                    --outdir ./test/jungwoo_1_attnloss_10/third_clip_skip \
                    --seed 42 \
                    --prompt "jw, white background, standing, smile, male_focused" --trg_token jw \
                    --H 512 --W 512 --clip_skip 2 \
















                    --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo.txt
################################################################################################################################################
python gen_img_diffusers.py \
                    --network_module networks.lora \
                    --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
                    --network_weights ./result/train_network_block_wise_20230920/lsy-000017.safetensors \
                    --outdir ./result/train_network_block_wise_20230920-epoch17 \
                    --seed 42 \
                    --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/test_style.txt

