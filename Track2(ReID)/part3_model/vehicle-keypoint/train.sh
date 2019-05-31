CUDA_VISIBLE_DEVICES=0,1,2,3 \
python pose_estimation/train.py \
    --cfg experiments/veri/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
