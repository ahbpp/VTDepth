# Exploring Efficiency of Vision Transformers for Self-Supervised Monocular Depth Estimation


The training and testing codes are based on  [Monodepth 2](https://github.com/nianticlabs/monodepth2).
We make this code more general. In this version, it is convenient to implement any model for self-supervised depth estimation. 
 
 
To download and prepare the KITTI dataset following [instructions](https://github.com/nianticlabs/monodepth2#-kitti-training-data).  
**Warning:** KITTI dataset is weighed about 175GB. 

## Training
 Training examples for all models are provided in `scripts` folder. 
 `configs` folder contains configs and parameters for experiments on KITTI.
 
 
 Scripts to train specific models:
 
| Model                     | Train |Abs. Rel.  | Train script                         |weights |
|---------------------------|-------|-----------|--------------------------------------|--------|
| VTDepthA0                 |  MS   |0.113      | `VTDepthA0_use_stereo.sh`            |[link](https://disk.yandex.ru/d/YY_o9BnrGB6h8Q)| 
| VTDepthB0                 |  MS   |0.108      | `VTDepthB0_use_stereo.sh`            |[link](https://disk.yandex.ru/d/8usQyZFQ_2CTCw)| 
| VTDepthB1                 |  MS   |0.101      | `VTDepthB1_use_stereo.sh`            |[link](https://disk.yandex.ru/d/OGLXekKcLKh0mg)| 
| VTDepthB2                 |  MS   |0.99       | `VTDepthB2_use_stereo.sh`            |[link](https://disk.yandex.ru/d/xrlhEqFfOBBS8A)| 
| VTDepthC0                 |  MS   |0.113      | `VTDepthC0_use_stereo.sh`            |[link](https://disk.yandex.ru/d/_CKabzoB8VPB6A)| 
| VTDepthB0                 |  M    |0.113      | `VTDepthB0_mono.sh`                  |[link](https://disk.yandex.ru/d/498TS3KkmmPXJg)| 
| VTDepthB1                 |  M    |0.109      | `VTDepthB1_mono.sh`                  |[link](https://disk.yandex.ru/d/gOxem-xgTB-Sww)| 
| VTDepthB2                 |  M    |0.105      | `VTDepthB2_mono.sh`                  |[link](https://disk.yandex.ru/d/UceoDiAKOJ4oCQ)| 
   

You can predict depth for a single image with:
`python test_simple.py --image_path test_image --no_cuda` 
Weights for VTDepthB0 locates in `depth_models_weights/VTDepthB0` folder. Weights for other models you can download
from links. 


## Evaluation
We use eigen split for evaluation.    
To prepare the ground truth depth maps run:

```buildoutcfg
python export_gt_depth.py --data_path kitti_data --split eigen
```

The next command evaluates VTDepthB0 model:  
```buildoutcfg
python evaluate_depth.py --config configs.pvt_v2_depth_resnet_pose_ln2_pretrained --load_weights_folder depth_models_weights/VTDepthB0 --eval_mono --data_path /path/to/Datasets/kitti_data --eval_split eigen --batch_size 10
```



