import networks
import torch
import torch.nn as nn
import torch.optim as optim
from transformer_networks.gvt import pcpvt_small_v0
import numpy as np
from functools import partial


def load_weights(encoder):
    try:
        path = "/path/to/models_weights/pcpvt_small.pth" #weights pretrained on imagenet
        #download from https://drive.google.com/file/d/1TWIx_8M-4y6UOKtbCgm1v-UVQ-_lYe6X/view?usp=sharing
        weights = torch.load(path, map_location="cpu")
        model_sd = encoder.state_dict()
        for name, weight in weights.items():
            if name in model_sd:
                if model_sd[name].dim() > 1 and model_sd[name].shape[1] != weights[name].shape[1]:
                    print(name, model_sd[name].shape, weights[name].shape)
                    to_load = torch.cat([weights[name]] * 2, dim=1)
                else:
                    to_load = weights[name]
                model_sd[name] = to_load
        encoder.load_state_dict(model_sd)
    except:
        print("No weights, pretrained on ImageNet")
    return encoder


def build_depth(img_size):
    encoder = pcpvt_small_v0(
        img_size=np.array(img_size))
    encoder = load_weights(encoder)
    decoder = networks.UpdatedDepthDecoder(encoder.embed_dims)
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



