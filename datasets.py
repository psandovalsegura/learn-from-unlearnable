import os
import pickle
import collections
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms, datasets

from constants import DATA_SETUPS, CIFAR10_ROOT, CIFAR100_ROOT, SVHN_ROOT, \
                      IMAGENET100_ROOT, TRANSFORM_OPTIONS, NORMALIZE_CONSTANTS

class ImageNetMini(datasets.ImageNet):
    def __init__(self, root, split='train', **kwargs):
        super(ImageNetMini, self).__init__(root, split=split, **kwargs)
        self.new_targets = []
        self.new_images = []
        for i, (file, cls_id) in enumerate(self.imgs):
            if cls_id <= 99:
                self.new_targets.append(cls_id)
                self.new_images.append((file, cls_id))
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs
        print('[class ImageNetMini] num samples:', len(self.samples))
        print('[class ImageNetMini] num targets:', len(self.targets))
        return
    
class NTGA(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        # x_file should be named with an 'x_train_' prefix
        # y_file should be named with a 'y_train_' prefix
        # choose file which matches the prefix
        x_file = [f for f in os.listdir(root) if f.startswith('x_train_')][0]
        y_file = [f for f in os.listdir(root) if f.startswith('y_train_')][0] # in one-hot encoding
        x = np.load(os.path.join(root, x_file))
        y = np.load(os.path.join(root, y_file))
        x = (x * 255).astype(np.uint8)
        y = np.argmax(y, axis=1)       # convert to index label
        self.data = x
        self.targets = y
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target 
    
    def __len__(self):
        return len(self.data)

class TAPFormatPoison(torch.utils.data.Dataset):
    """
    A poison format consisting of a data/ folder with images named
    {base_idx}.png, where base_idx is the index of the image in the
    base dataset.
    """
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        _, label = self.baseset[idx]
        return self.transform(Image.open(os.path.join(self.root, 'data',
                                            f'{idx}.png'))), label
    
class TAPFormatPoisonOrthoProj(torch.utils.data.Dataset):
    """
    A poison format consisting of a data/ folder with images named
    {base_idx}.png, where base_idx is the index of the image in the
    base dataset.
    """
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = sorted(os.listdir(os.path.join(root, 'data')))
        self.root = root
        self._perturb_images()

    def _perturb_images(self):
        self.clean_images = []
        self.images = []
        self.labels = []
        for idx in range(len(self.baseset)):
            base_idx = int(self.samples[idx].split('.')[0])
            clean_image, label = self.baseset[base_idx]
            image = np.array(Image.open(os.path.join(self.root, 'data',
                                            self.samples[idx])))
            self.images.append(image)
            self.labels.append(label)
            self.clean_images.append(clean_image)

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]

class UnlearnablePoison(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name, baseset):
        self.baseset = baseset
        self.dataset_name = dataset_name
        self.root = root
        self.classwise = 'classwise' in root
        noise = torch.load(os.path.join(root, 'perturbation.pt'))

        # Load images into memory to prevent IO from disk
        self._perturb_images(noise)

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.baseset[idx]

    def _perturb_images(self, noise):
        if self.dataset_name in ['STL10', 'SVHN']:
            perturb_noise = noise.mul(255).clamp_(-255, 255).to('cpu').numpy()
        else:
            perturb_noise = noise.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.baseset.data = self.baseset.data.astype(np.float32)
        for i in range(len(self.baseset)):
            if self.classwise:
                if self.dataset_name in ['STL10', 'SVHN']:
                    self.baseset.data[i] += perturb_noise[self.baseset.labels[i]]
                else:
                    self.baseset.data[i] += perturb_noise[self.baseset.targets[i]]
            else: # samplewise
                self.baseset.data[i] += perturb_noise[i]
            self.baseset.data[i] = np.clip(self.baseset.data[i], a_min=0, a_max=255)
        self.baseset.data = self.baseset.data.astype(np.uint8)

class UnlearnableImageNetPoison(ImageNetMini):
    def __init__(self, root, split, poison_rate=1.0, seed=0,
                 perturb_tensor_filepath=None, **kwargs):
        super(UnlearnableImageNetPoison, self).__init__(root=root, split=split, **kwargs)
        np.random.seed(seed)
        self.poison_rate = poison_rate
        self.perturb_tensor = torch.load(perturb_tensor_filepath)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()

        # Random Select Poison Targets
        targets = list(range(0, len(self)))
        self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True

        print('[class UnlearnableImageNetPoison] Perturb tensor shape:', self.perturb_tensor.shape)
        print('[class UnlearnableImageNetPoison] Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = np.array(transforms.RandomResizedCrop(224)(sample)).astype(np.float32)

        if self.poison_samples[index]:
            noise = self.perturb_tensor[target]
            sample = sample + noise
            sample = np.clip(sample, 0, 255)
        sample = sample.astype(np.uint8)
        sample = Image.fromarray(sample).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class RobustErrorMin(torch.utils.data.Dataset):
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.root = root
        
        with open(root, 'rb') as f:
            raw_noise = pickle.load(f)
            self._perturb_images(raw_noise)

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.baseset[idx]
    
    def _perturb_images(self, raw_noise):
        assert isinstance(raw_noise, np.ndarray)
        assert raw_noise.dtype == np.int8

        raw_noise = raw_noise.astype(np.int16)

        noise = np.zeros_like(raw_noise)
        indices = np.random.permutation(len(noise))
        noise[indices] += raw_noise[indices]

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0,2,3,1])

        ''' add noise to images (uint8, 0~255) '''
        imgs = self.baseset.data.astype(np.int16) + noise
        imgs = imgs.clip(0,255).astype(np.uint8)
        self.baseset.data = imgs

class OPSDataset(torch.utils.data.Dataset):

    def __init__(self, data, perturbation, target, transform, pert=1) -> None:
        super().__init__()

        '''Clean Examples if 'pert' is False'''
    
        '''data format: np.ndarray, float32 range from 0 to 1, H x W x C'''

        self.data = data
        self.perturbation = perturbation
        self.target = target
        self.transform = transform
        self.pert = pert

        '''Perturbation mode: S for sample-wise, C for class-wise, U for universal'''

        if len(self.perturbation.shape) == 4:
            if self.perturbation.shape[0] == len(self.target):
                self.mode = 'S'
            else:
                self.mode = 'C'
        else:
            self.mode = 'U'

    def __len__(self):

        return len(self.target)

    def __getitem__(self, index:int):
        
        if self.pert == 1:
            if self.mode == 'S':
                img_p, target = self.data[index] + self.perturbation[index], self.target[index]
            elif self.mode == 'C':
                img_p, target = self.data[index] + self.perturbation[self.target[index]], self.target[index]
            else:
                img_p, target = self.data[index] + self.perturbation, self.target[index]

        elif self.pert == 2:
            img_p, target = self.perturbation[index], self.target[index]
            
        else:
            img_p, target = self.data[index], self.target[index]

        img_p = np.clip(img_p, 0, 1)
        img_p = np.uint8(img_p * 255)
        img_p = Image.fromarray(img_p)
        
        if self.transform is not None:
            img_p = self.transform(img_p)
            
        return img_p, target

class LSPDataset(torch.utils.data.Dataset):
    def __init__(self, root, baseset, num_classes=10):
        """
        Dataset class for Linearly Separable Poison (LSP) from 
        "Availability Attacks Create Shortcuts" by Yu et al., 2022
        """
        self.baseset = baseset
        self.root = root
        self.num_classes = num_classes
        
        # load perturbation
        lsp_perturbations = np.load(os.path.join(root, 'simple_data.npy'))
        lsp_labels = np.load(os.path.join(root, 'simple_label.npy'))
        self._perturb_images(lsp_perturbations, lsp_labels)

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.baseset[idx]
    
    def _perturb_images(self, lsp_perturbations, lsp_labels):
        self.baseset.data = self.baseset.data.astype(np.float)/255.
        arr_target = np.array(self.baseset.targets)
        # add synthetic noises to original examples
        for label in range(self.num_classes):
            orig_data_idx = arr_target == label
            simple_data_idx = lsp_labels == label
            mini_simple_data = lsp_perturbations[simple_data_idx][0:int(sum(orig_data_idx))]
            self.baseset.data[orig_data_idx] += mini_simple_data

        self.baseset.data = np.clip((self.baseset.data*255), 0, 255).astype(np.uint8)

class PoisonWithCleanDataset(torch.utils.data.Dataset):
    def __init__(self, poison_ds, baseset):
        self.poison_ds = poison_ds
        self.baseset = baseset

    def __len__(self):
        return len(self.baseset)
    
    def __getitem__(self, idx):
        clean_img, label = self.baseset[idx]
        poison_img, _ = self.poison_ds[idx]
        return poison_img, label, clean_img

def get_standard_dataset(dataset_name, train, transforms_key, normalize=True, transform_only=False):
    transform_list = TRANSFORM_OPTIONS[dataset_name][transforms_key].copy()
    normalize_mean = NORMALIZE_CONSTANTS[dataset_name]['mean'].copy()
    normalize_std = NORMALIZE_CONSTANTS[dataset_name]['std'].copy()
    if dataset_name == 'CIFAR10':
        if normalize:
            transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
        transform = transforms.Compose(transform_list)
        if transform_only:
            return None, transform
        ds = datasets.CIFAR10(root=CIFAR10_ROOT, train=train, download=False, transform=transform)
    elif dataset_name == 'CIFAR100':
        if normalize:
            transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
        transform = transforms.Compose(transform_list)
        if transform_only:
            return None, transform
        ds = datasets.CIFAR100(root=CIFAR100_ROOT, train=train, download=False, transform=transform)
    elif dataset_name == 'SVHN':
        if normalize:
            transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
        transform = transforms.Compose(transform_list)
        if transform_only:
            return None, transform
        ds = datasets.SVHN(root=SVHN_ROOT, split='train' if train else 'test', download=False, transform=transform)
    elif dataset_name == 'IMAGENET100':
        if normalize:
            transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
        transform = transforms.Compose(transform_list)
        if transform_only:
            return None, transform
        ds = ImageNetMini(root=IMAGENET100_ROOT, split=('train' if train else 'val'), transform=transform)
    else:
        raise NotImplementedError
    return ds

def construct_train_dataset(setup_key, normalize, transforms_key=None, is_ortho_proj=False):
    """
    setup_key: key for dataset setup in DATA_SETUPS
    """
    setup = DATA_SETUPS[setup_key].copy()
    dataset_name, dataset_root, dataset_type = setup['dataset_name'], setup['root'], setup['dataset_type']
    transforms_key = 'train_transform' if transforms_key is None else transforms_key
    if dataset_type == 'STANDARD':
        ds = get_standard_dataset(dataset_name, train=True, transforms_key=transforms_key, 
                                  normalize=normalize)
    elif dataset_type == 'TAP':
        baseset = get_standard_dataset(dataset_name, train=True, transforms_key=transforms_key, 
                                       normalize=normalize)
        if is_ortho_proj:
            ds = TAPFormatPoisonOrthoProj(dataset_root, baseset)
        else:
            ds = TAPFormatPoison(dataset_root, baseset)
    elif dataset_type == 'ULE':
        baseset = get_standard_dataset(dataset_name, train=True, transforms_key=transforms_key, 
                                       normalize=normalize)
        ds = UnlearnablePoison(dataset_root, dataset_name, baseset)
    elif dataset_type == 'ULE_IMAGENET':
        _, transform = get_standard_dataset(dataset_name, train=True, transforms_key=transforms_key,
                                            normalize=normalize, transform_only=True)
        perturb_tensor_filepath = os.path.join(dataset_root, 'perturbation.pt')
        ds = UnlearnableImageNetPoison(root=IMAGENET100_ROOT, split='train', 
                                       perturb_tensor_filepath=perturb_tensor_filepath, 
                                       transform=transform)
    elif dataset_type == 'REM':
        baseset = get_standard_dataset(dataset_name, train=True, transforms_key=transforms_key, 
                                       normalize=normalize)
        ds = RobustErrorMin(dataset_root, baseset)
    elif dataset_type == 'NTGA':
        _, transform = get_standard_dataset(dataset_name, train=True, transforms_key=transforms_key, 
                                            normalize=normalize, transform_only=True)
        ds = NTGA(root=dataset_root, transform=transform)
    elif dataset_type == 'OPS':
        baseset = get_standard_dataset(dataset_name, train=True, transforms_key=transforms_key,
                                       normalize=normalize)
        baseset.targets = np.array(baseset.targets)
        perturbation = np.load(os.path.join(dataset_root, 'perturbation.npy'))
        ds = OPSDataset(data=(baseset.data / 255), 
                        perturbation=perturbation, 
                        target=baseset.targets, 
                        transform=baseset.transform)
    elif dataset_type == 'LSP':
        baseset = get_standard_dataset(dataset_name, train=True, transforms_key=transforms_key,
                                       normalize=normalize)
        ds = LSPDataset(root=dataset_root, baseset=baseset)
    else:
        raise NotImplementedError
    return ds


def get_train_dataset(setup_key, batch_size, num_workers, normalize=True, shuffle=True):    
    ds = construct_train_dataset(setup_key, normalize=normalize)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def get_test_dataset(setup_key, batch_size, num_workers, normalize=True):
    setup = DATA_SETUPS[setup_key].copy()
    dataset_name = setup['dataset_name']
    ds = get_standard_dataset(dataset_name, train=False, transforms_key='test_transform', normalize=normalize)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader