image_dir="" # dir of input testing images
masks_dir="" # dir of the keypoint heatmap of testing images generated from vehicle-keypoint/infer.sh
image_list="" # list of image names
model_dir="" # final reid model
output_dir="" # dir of output file
output_pkl="" # name of output file

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/infer.py \
    --image-dir $image_dir \
    --masks-dir $masks_dir \
    --input-list $image_list \
    --output-dir $output_dir \
    --output-pkl $output_pkl \
    -a aicity_masks_seresnext101 \
    --model-type 'masks' \
    --weights $model_dir/model_best.pth.tar \
    --num-m-features 128 \
    --masks 20 \
    --height 288 \
    --width 384 \
    --batch-size 32 \
