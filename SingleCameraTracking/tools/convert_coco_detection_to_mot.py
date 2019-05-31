import os
import json

dt_file = '../detection_res/aicity2019_track_train.pkl.json' 
#dt_file = '../detection_res/aicity2019_track_test.pkl.json' 
if 'train' in dt_file:
    dataset = 'train'
elif 'test' in dt_file:
    dataset = 'test'
else:
    print('error')
    exit()
mapping_file = '../detection_res/'+dataset+'.json' 

fp = open(dt_file, 'r')
dets = json.load(fp)
fp.close()
fp = open(mapping_file)
info = json.load(fp)
fp.close()

id_filename = {}
det_all = {}
for iminfo in info["images"]:
    id_filename[iminfo["id"]] = iminfo["file_name"]
for det in dets:
    id = det["image_id"]
    filename = id_filename[id]
    score = det["score"]
    bbox = det["bbox"]
    label = det["category_id"]

    filename = filename.split('/')
    scene = filename[2]
    cam = filename[3]
    frame = int(filename[5].split('.')[0])
    
    if scene not in det_all:
        det_all[scene] = {}
    if cam not in det_all[scene]:
        det_all[scene][cam] = {}
    if frame not in det_all[scene][cam]:
        det_all[scene][cam][frame] = []
    det_all[scene][cam][frame].append(det)

for scene in det_all:
    for cam in det_all[scene]:
        frame_list = det_all[scene][cam]
        fpath = os.path.join('../aic19/aic19-track1-mtmc',dataset, scene, cam, 'det/det_se154_csc_ms.txt')
        fp = open(fpath, 'w')
        for frame in frame_list:
            for det_out in frame_list[frame]:
                bbox = det_out["bbox"]
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                score = det_out["score"]
                label = det_out["category_id"]
                fp.write('%d,-1,%.3f,%.3f,%.3f,%.3f,%.5f,-1,-1,-1\n' % (frame,x,y,w,h,score))
        fp.close()
