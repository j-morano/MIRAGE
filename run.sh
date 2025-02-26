
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
    # --infer_only --test \
    # --finetune './__weights/MIRAGE-Base.pth' \
    # --finetune './__weights/MIRAGE-Large.pth' \
    # --finetune './__weights/medsam_vit_b.pth' \
    # --finetune './__weights/RETFound_vit-l_bscan_weights.pth' \
    # --finetune './__weights/dinov2_vitl14_pretrain.pth' \
