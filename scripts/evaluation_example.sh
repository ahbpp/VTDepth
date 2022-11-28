python evaluate_depth.py --config configs.pvt_v2_depth_resnet_pose_ln2_pretrained \
--load_weights_folder /path/to/TrainLogs/pvt_v2_depth_resnet_pose_ln2_pretrained_use_stereo/models/last_weights \
--eval_mono --data_path /path/to/Datasets/kitti_data --eval_split eigen --batch_size 10