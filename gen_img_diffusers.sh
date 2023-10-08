cuda_visible_devices=2 python generate_crossattn_map.py \
                    --network_module networks.lora \
                    --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
                    --network_weights "./result/jungwoo_experience/jungwoo_4_crossattn_calculate_change_only_up/jungwoo-000018.safetensors" \
                    --outdir ./test/jungwoo_1_base/third_clip_skip \
                    --seed 42 \
                    --prompt "jw, white background, standing, smile, male_focused" --trg_token jw \
                    --H 512 --W 512 --clip_skip 2 \
                    # --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo.txt

      --prompt "jw, white_background" --outdir 20231008_result/jungwoo_4_crossattn_calculate_change_only_up --seed 42 --trg_token 'jw'