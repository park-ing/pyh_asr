CUDA_VISIBLE_DEVICES=0 python preprocess_lrs2lrs3.py \
    --data-dir '/home/nas4/DB/DB_AVSR/original/LRS3' \
    --landmarks-dir '/home/nas4/DB/DB_AVSR/original/LRS3/LRS3_landmarks' \
    --root-dir '/home/nas4/DB/DB_AVSR/pyh_avsr' \
    --dataset 'lrs3' \
    --subset 'train' \
    --seg-duration 4 \
    --groups 1 \
    --job-index 0

    /home/nas4/DB/AIHUB_track2_2/data