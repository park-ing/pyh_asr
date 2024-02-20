CUDA_VISIBLE_DEVICES=0 python preprocess_kor.py \
    --data-dir '/home/nas4/DB/AIHUB_track2_2/data' \
    --root-dir '/home/nas4/DB/AIHUB_track2_2/pyh' \
    --dataset 'Tr2DZ' \
    --subset 'train' \
    --seg-duration 24 \
    --groups 1 \
    --job-index 0