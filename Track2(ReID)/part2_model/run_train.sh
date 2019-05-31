export CUDA_VISIBLE_DEVICES=0,1,2,3 
python examples/train.py \
    -a HighResolutionNet_reid \
    -b 32 \
    -d gao_crop_train \
    --combine-trainval \
    --frozen_sublayer True \
    --weights ./weights/hrnetv2_w32_imagenet_pretrained.pth \
    --logs-dir ./logs/model_288_384/ \
    --lr 0.01 \
    --gamma 0.1 \
    --weight-decay 0.0005 \
    --warm_up_factor 0.01 \
    --warm_up_iter 100 \
    --step_epoch 600 900 1000 \
    --epochs 1100 \
    --lr_mult 1.0 \
    --metric_loss_weight 0.5 \
    --big_height 288 \
    --big_width 384 \
    --target_height 288 \
    --target_width 384 \
    --epoch_inter 1 \
    --start_save 50 \
    --dense_evaluate 1090 \
    --num_instances 4 \

