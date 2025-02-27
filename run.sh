
# Classification
./runner python run_cls_tuning.py \
    --runners 1 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --linear_probing \
    --data_root \
        ~/tmp/MIRAGE_DATASETS/Classification \
    --data_set \
        GAMMA
exit


# Segmentation
./runner python run_seg_tuning.py \
    --runners 1 \
    -- \
    --version v1 \
    --config \
        ./_cfgs/ft_semseg_200e_convnext.yaml \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --data_path \
        ~/tmp/MIRAGE_DATASETS/Segmentation/Duke_DME/
exit
