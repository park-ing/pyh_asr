CUDA_VISIBLE_DEVICES=0,1 python infer.py data.modality=audio \
                ckpt_path=/home/nas4/user/yh/ai_hub_korean/outputs/2023-10-19/22-58-25/Tr2DZ/mid_V1_zeroshot/model_avg_5.pth\
                trainer.num_nodes=1 \
                infer_path=/home/nas4/user/yh/ai_hub_korean/test.txt

# mod v1
# /home/nas4/user/yh/ai_hub_korean/outputs/2023-10-19/22-58-25/Tr2DZ/mid_V1_zeroshot/model_avg_5.pth