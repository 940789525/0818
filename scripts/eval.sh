# ViT-B/32
job_name="CCVTR_msvd_vit32_32_DL_0725_1_eavl"
DATA_PATH="/home/wa24301158/dataset/MSVD"

# 设置环境变量，指定使用 GPU 3
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 \
    main_xclip.py --do_eval --num_thread_reader=4 \
    --epochs=5 --batch_size=48 --n_display=5 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/msvd_hevc \
    --mask_path ${DATA_PATH}/videos_hevc_info \
    --output_dir ckpts3/${job_name} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 9 --slice_framepos 0 \
    --loose_type --linear_patch 2d --sim_header meanP \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a logs/${job_name}