#!/bin/bash
#TransEncCNNDec(VTDepthB0); pretrained on imagenet; DataType MS(mono video and stereo pairs)

python3 train.py --data_path /path/to/Datasets/kitti_data \
--log_dir /path/to/TrainLogs  --model_name pvt_v2_depth_resnet_pose_ln2_pretrained \
--split eigen_zhou --num_workers 8 --eval_mono --pose_model_type separate_resnet \
--learning_rate 1e-4 --num_epochs 30 --scheduler_step_size 21 \
--config configs.pvt_v2_depth_resnet_pose_ln2_pretrained --batch_size 12 --use_stereo