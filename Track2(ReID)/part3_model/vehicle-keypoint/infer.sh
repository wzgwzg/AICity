input_dir="" # input training/testing images
image_list="" # list of image names
output_dir="" # output dir of the keypoint heatmap of input images

# for example
# input_dir="data/aicity_keypoint/images"
# image_list="./images.txt"
# output_dir="../vehicle-reid-keypoint/examples/data/small_vehicle/masks"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python pose_estimation/infer.py \
    --cfg experiments/veri/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --gpus 0,1,2,3 \
    --batch-size 64 \
    --input-dir $input_dir \
    --input-list $image_list \
    --model-file model_best.pth.tar \
    --output-dir $output_dir \
