CUDA_VISIBLE_DEVICES=0 python preprocess_lrs2lrs3.py \
    --data-dir '/home/nas4/DB/DB_AVSR/original/LRS2_src' \
    --landmarks-dir '/home/nas4/DB/DB_AVSR/original/LRS2_src/LRS2_landmarks' \
    --root-dir '/home/nas4/DB/DB_AVSR/pyh_avsr' \
    --dataset 'lrs2' \
    --subset 'test' \
    --seg-duration 24 \
    --groups 1 \
    --job-index 0