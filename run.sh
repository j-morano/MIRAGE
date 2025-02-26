
# Classification

python3 run_cls_tuning.py \
    --seed 0 \
    --model \
        RETFound \
    --linear_probing \
    --early_stopping_epochs 20 \
    --early_start_from 20 \
    --val_metric bacc \
    --early_stopping_delta 0.001 \
    --val_metric_two loss \
    --early_stopping_delta_two 0.001 \
    --version VLM_bacc-loss_20-20 \
    --num_workers 8 \
    --pool token_mix \
    --data_set \
        GAMMAv2
exit



# Segmentation

./runner python3 run_seg_tuning.py \
    --runners 1 \
    -- \
    --config \
        ./_cfgs/ft_semseg_200e_convnext.yaml \
    --finetune \
        './__weights/MIRAGE-Base.pth' \
    --data_path '/home/morano/tmp/MIRAGE_DATASETS/Segmentation/Duke_DME/' \
    --version v1
exit
