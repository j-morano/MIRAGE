
source ./venv/bin/activate


# Pretraining
./runner python run_pretraining.py \
    --runners 1 \
    -- \
    --config ./_cfgs/pre_mirage_98_1600e_bscan-slo-bscanlayermap_512-128--32-8.yaml \
    --data_path \
        ./__datasets/Pretraining/ \
    --weights \
        ./__weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth \
        ./__weights/mae_pretrain_vit_large.pth
exit


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


# Segmentation tuning
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


# Segmentation evaluation
./runner python eval_seg.py \
    --runners 1 \
    -- \
    --model_path \
        ./__output/seg/v1/Duke_DME/MIRAGE-Base_frozen_convnext_CEGDice/ \
        ./__output/seg/v1/Duke_DME/MIRAGE-Large_frozen_convnext_CEGDice/
exit
