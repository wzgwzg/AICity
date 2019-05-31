CUDA_VISIBLE_DEVICES=12,13 \
python examples/multi_attribute.py \
    -a multi_attribute_3_resnet50 \
    -b 64 \
    -d aicity_attribute \
    --combine-trainval \
    --weights ./pretrain_models/resnet50-19c8e357.pth \
    --logs-dir ./logs/adam_new_attribute/ \
    --optimizer 1 \
    --lr 0.01 \
    --weight-decay 0.0005 \
    --epochs 800 \
    --step_size 400 \
    --step_size2 700 \
    --step_size3 750 \
    --lr_mult 1.0 \
    --metric_loss_weight 0.02 \
    --big_height 310 \
    --big_width 414 \
    --target_height 288 \
    --target_width 384 \
    --epoch_inter 100 \
    --start_save 0 \
    --dense_evaluate 800 \
    --warm_up_ep 100 \
    --is_cls 1 \

