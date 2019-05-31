CUDA_VISIBLE_DEVICES=8,9 \
python examples/test_extract_attribute.py \
    -a multi_attribute_3_resnet50 \
    -b 32 \
    -d aicity_attribute \
    --combine-trainval \
    --weights ./logs/attribute_model/the_trained_model_weights_file \
    --logs-dir ./logs/attribute_model \
    --lr 0.01 \
    --weight-decay 0.0005 \
    --epochs 100 \
    --step_size 70 \
    --lr_mult 1.0 \
    --metric_loss_weight 0.02 \
    --height 288 \
    --width 384 \
