import torch.nn as nn
import torch.nn.functional as F
from models import *
from torchvision import models as torch_models
from constants import DATA_SETUPS, NUM_CLASSES

def get_model(model_name, setup_key):
    setup = DATA_SETUPS[setup_key].copy()
    dataset_name = setup['dataset_name']
    num_classes = NUM_CLASSES[setup['dataset_name']]

    # ImageNet models require different architectures due to input dimensions
    if 'IMAGENET' in dataset_name:
        if model_name == 'resnet18':
            model = torch_models.resnet18(pretrained=False, num_classes=num_classes).cuda()
        elif model_name == 'advprop-resnet18':
            model = torch_models.resnet18(pretrained=False, 
                                          norm_layer=MixBatchNorm2d,
                                          num_classes=num_classes).cuda()
        else:
            raise NotImplementedError(f'ImageNet models not implemented for {model_name}!')
        return model

    if model_name == 'resnet18':
        model = ResNet18(num_classes=num_classes).cuda()
    elif model_name == 'advprop-resnet18':
        model = AdvPropResNet18(num_classes=num_classes).cuda()
    elif model_name == 'densenet121':
        model = DenseNet121(num_classes=num_classes).cuda()
    elif model_name == 'vgg16':
        model = VGG('VGG11').cuda()
    elif model_name == 'vgg19':
        model = VGG('VGG19').cuda()
    elif model_name == 'googlenet':
        model = GoogLeNet().cuda()
    elif model_name == 'vit-patch-size-4':
        model = ViT(image_size = 32,
                    patch_size = 4,
                    num_classes = 10,
                    dim = 384,
                    depth = 7,
                    heads = 12,
                    mlp_dim = 384,
                    dropout = 0.0,
                    emb_dropout = 0.0).cuda()
    elif model_name == 'vit-patch-size-8':
        model = ViT(image_size = 32,
                    patch_size = 8,
                    num_classes = 10,
                    dim = 384,
                    depth = 7,
                    heads = 12,
                    mlp_dim = 384,
                    dropout = 0.0,
                    emb_dropout = 0.0).cuda()
    else:
        raise ValueError(f'Unknown model name {model_name}!')
    return model

def initialize_checkpoint(model_name, setup_key, ckpt_path):
    model = get_model(model_name, setup_key=setup_key)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict=state_dict)
    model.cuda()
    model.eval()
    return model
