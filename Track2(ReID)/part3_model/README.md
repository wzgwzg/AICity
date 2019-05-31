# Vehicle Reid using Keypoints

## Pipeline

### Data
We labeled keypoints on the training data of track2 following the definition of vehicle keypoint in "Orientation Invariant Feature Embedding and Spatial Temporal Regularization for Vehicle Re-identification". The labeled data are keypoint_train.txt(for training)/keypoint_test.txt(for validation) in vehicle-keypoint/data/aicity_keypoint. Note that the key point annotation corresponds to the renamed aicity training set which can be found in complete_aicity_car196.tar.gz mentioned in the main README file.  

### Training
1. run vehicle-keypoint/train.sh to train vehicle keypoint model.
2. run vehicle-keypoint/infer.sh to generate heatmap of training data.
3. run vehicle-reid-keypoint/train.sh to train vehicle-reid model.

### Testing
1. run vehicle-keypoint/infer.sh to generate heatmap of testing data.
2. run vehicle-reid-keypoint/infer.sh to generate reid features for retrieval.
