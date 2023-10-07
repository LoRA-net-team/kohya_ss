cuda_visible_devices=2 python generate_crossattn_map.py \
                    --network_module networks.lora \
                    --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
                    --network_weights "./result/jungwoo_experience/jungwoo_1_base/jungwoo-000017.safetensors" \
                    --outdir ./test/jungwoo_1_base/third_clip_skip \
                    --seed 42 \
                    --prompt "jw, white background, standing, smile, male_focused" --trg_token jw \
                    --H 512 --W 512 --clip_skip 2 \
                    # --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo.txt