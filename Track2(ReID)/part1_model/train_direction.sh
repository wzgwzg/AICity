CUDA_VISIBLE_DEVICES=14,15 \
python examples/train_direction.py \
    -a direction_resnet50 \
    -b 32 \
    --weights ./pretrain_models/resnet50-19c8e357.pth \
    --logs-dir ./logs/direction_output/ \
    --optimizer 1 \
    --lr 1e-3 \
    --weight-decay 0.0005 \
    --epochs 20 \
    --step_size 50 \
    --step_size2 70 \
    --step_size3 90 \
    --lr_mult 1.0 \
    --big_height 310 \
    --big_width 414 \
    --target_height 288 \
    --target_width 384 \
    --epoch_inter 2 \
    --start_save 0 \
    --dense_evaluate 100 \
    --warm_up_ep 10 \

