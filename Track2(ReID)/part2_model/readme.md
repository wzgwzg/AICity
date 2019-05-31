This part contains the source codes of training three vehicle reid models including resnet50_sac, hrnet and MGN. As for feature extraction, related codes in 'part1_model' can be used.

## preparing dataset
The dataset defination is the same as the one in [open-reid](https://github.com/Cysu/open-reid).
Users should define their own dataset in the './reid/datasets'
The data should be put into the folder './examples/data'

## training
```
sh run.sh
```
The example code trains the resnet50_sac network (which is proposed in [self-attention learning for person re-identification](http://www.bmva.org/bmvc/2018/contents/papers/0613.pdf)), other supported networks can be found in the './reid/models' folder.
The meanings of key parameters can be found in the './examples/train.py'
