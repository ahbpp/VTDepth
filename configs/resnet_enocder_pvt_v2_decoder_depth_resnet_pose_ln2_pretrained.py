import networks
import torch
import torch.nn as nn
import torch.optim as optim
from transformer_networks.pvtv2 import PVTDecoderV2V2Res
import numpy as np
from functools import partial



def build_depth(img_size):
    encoder = networks.ResnetEncoder(18, True)
    decoder = PVTDecoderV2V2Res(
        img_size=img_size,
        depths=[1, 1, 1, 1],
        embed_dim=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[8, 8, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0, drop_path_rate=0.1
    )
    return encoder, decoder


def build_models(opt, device):
    models = {}
    depth_encoder, depth_decoder = build_depth(img_size=[opt.height, opt.width])
    models["encoder"] = depth_encoder.to(device)
    models["depth"] = depth_decoder.to(device)
    num_input_frames = len(opt.frame_ids)
    num_pose_frames = 2
    if opt.pose_model_type == "separate_resnet":
        models["pose_encoder"] = networks.ResnetEncoder(
            opt.num_layers,
            opt.weights_init == "pretrained",
            num_input_images=num_pose_frames).to(device)


        models["pose"] = networks.PoseDecoder(
            models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2).to(device)

    elif opt.pose_model_type == "shared":
        models["pose"] = networks.PoseDecoder(
            models["encoder"].num_ch_enc, num_pose_frames).to(device)

    elif opt.pose_model_type == "posecnn":
        models["pose"] = networks.PoseCNN(
            num_input_frames if opt.pose_model_input == "all" else 2).to(device)
    return models


def build_optim(models, opt):
    parameters = []
    parameters += list(models["encoder"].parameters())
    parameters += list(models["depth"].parameters())
    parameters += list(models["pose_encoder"].parameters())
    parameters += list(models["pose"].parameters())
    model_optimizer = optim.Adam(parameters, opt.learning_rate)
    return model_optimizer

def build_scheduler(model_optimizer, opt):
    model_lr_scheduler = optim.lr_scheduler.StepLR(
        model_optimizer, opt.scheduler_step_size, 0.1)
    return model_lr_scheduler



