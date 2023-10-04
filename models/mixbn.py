import torch
import torch.nn as nn

class MixBatchNorm2d(nn.BatchNorm2d):
    '''
    Adapted from: https://github.com/tingxueronghua/pytorch-classification-advprop/blob/master/imagenet.py
    
    If the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].
    If you set batch_type as 'mix', this network will use one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    
    During training, the batch_type should be set as 'mix'.
    During validation, we only need the results of the features using some specific batch normalization.
    If you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.
    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.   
    Usage:
        # Function definitions
        def to_status(m, status):
            if hasattr(m, 'batch_type'):
                m.batch_type = status
        to_clean_status = partial(to_status, status='clean')
        to_adv_status = partial(to_status, status='adv')
        to_mix_status = partial(to_status, status='mix')
        # Model application
        model.apply(to_clean_status)
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input