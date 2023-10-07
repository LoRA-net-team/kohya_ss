cuda_visible_devices=2 python crossattnmap.py \
                    --network_module networks.lora \
                    --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
                    --network_weights "./result/jungwoo_experience/jungwoo_1_base/jungwoo-000017.safetensors" \
                    --outdir ./test \
                    --seed 42 \
                    --prompt "jw, white background" --trg_token jw \
                    --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo.txt
################################################################################################################################################
python gen_img_diffusers.py \
                    --network_module networks.lora \
                    --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
                    --network_weights ./result/train_network_block_wise_20230920/lsy-000017.safetensors \
                    --outdir ./result/train_network_block_wise_20230920-epoch17 \
                    --seed 42 \
                    --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/test_style.txt

