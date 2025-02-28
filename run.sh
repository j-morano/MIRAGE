
source ./venv/bin/activate


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
        ./__datasets/Classification \
    --data_set \
        GAMMA
exit


# Segmentation
./runner python run_seg_tuning.py \
    --runners 1 \
    -- \
    --version v1 \
    --config \
        ./_cfgs/seg_200e_convnext.yaml \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --data_path \
        ./__datasets/Segmentation/Duke_DME/
exit
