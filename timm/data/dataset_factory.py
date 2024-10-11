""" Dataset Factory

Hacked together by / Copyright 2021, Ross Wightman
"""
import os

from torchvision.datasets import CIFAR100, CIFAR10, MNIST, KMNIST, FashionMNIST, ImageFolder
try:
    from torchvision.datasets import Places365
    has_places365 = True
except ImportError:
    has_places365 = False
try:
    from torchvision.datasets import INaturalist
    has_inaturalist = True
except ImportError:
    has_inaturalist = False
try:
    from torchvision.datasets import QMNIST
    has_qmnist = True
except ImportError:
    has_qmnist = False
try:
    from torchvision.datasets import ImageNet
    has_imagenet = True
except ImportError:
    has_imagenet = False

from .dataset import IterableImageDataset, ImageDataset, MCImageDataset, SwitchmageDataset

_TORCH_BASIC_DS = dict(
    ImageNet=ImageNet,
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    kmnist=KMNIST,
    fashion_mnist=FashionMNIST,
)
_TRAIN_SYNONYM = dict(train=None, training=None)
_EVAL_SYNONYM = dict(val=None, valid=None, validation=None, eval=None, evaluation=None)


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root


def create_dataset(
        name,
        root,
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=False,
        batch_size=None,
        seed=42,
        repeats=0,
        img_mode='RGB',
        corrupted=None,
        post_dwt=False,
        dataset_alias="imagenet",
        enable_aux=False,
        switch_aux=-1,
        sdm = [0,0,0,0],
        **kwargs
):
    name = name.lower()
    if name.startswith('torch/'):
        name = name.split('/', 2)[-1]
        torch_kwargs = dict(root=root, download=download, **kwargs)
        
        if name in _TORCH_BASIC_DS:
            assert has_imagenet , 'Please update for Imagenet in torchvision!!! '
            ds_class = _TORCH_BASIC_DS[name]
            use_train = split in _TRAIN_SYNONYM
            print('TAG => {}'.format(torch_kwargs))
            ds = ds_class(train=use_train, **torch_kwargs)
        elif name == 'inaturalist' or name == 'inat':
            assert has_inaturalist, 'Please update to PyTorch 1.10, torchvision 0.11+ for Inaturalist'
            target_type = 'full'
            split_split = split.split('/')
            if len(split_split) > 1:
                target_type = split_split[0].split('_')
                if len(target_type) == 1:
                    target_type = target_type[0]
                split = split_split[-1]
            if split in _TRAIN_SYNONYM:
                split = '2021_train'
            elif split in _EVAL_SYNONYM:
                split = '2021_valid'
            ds = INaturalist(version=split, target_type=target_type, **torch_kwargs)
        elif name == 'places365':
            assert has_places365, 'Please update to a newer PyTorch and torchvision for Places365 dataset.'
            if split in _TRAIN_SYNONYM:
                split = 'train-standard'
            elif split in _EVAL_SYNONYM:
                split = 'val'
            ds = Places365(split=split, **torch_kwargs)
        elif name == 'qmnist':
            assert has_qmnist, 'Please update to a newer PyTorch and torchvision for QMNIST dataset.'
            use_train = split in _TRAIN_SYNONYM
            ds = QMNIST(train=use_train, **torch_kwargs)
        elif name == 'imagenet':
            assert has_imagenet, 'Please update to a newer PyTorch and torchvision for ImageNet dataset.'
            if split in _EVAL_SYNONYM:
                split = 'val'
            #print('TAG => {}'.format(torch_kwargs))
            del torch_kwargs['download']
            ds = ImageNet(split=split, **torch_kwargs)
        elif name == 'image_folder' or name == 'folder':
            # in case torchvision ImageFolder is preferred over timm ImageDataset for some reason
            if search_split and os.path.isdir(root):
                # look for split specific sub-folder in root
                root = _search_split(root, split)
            ds = ImageFolder(root, **kwargs)
        else:
            assert False, f"Unknown torchvision dataset {name}"
    elif name.startswith('hfds/'):
        # NOTE right now, HF datasets default arrow format is a random-access Dataset,
        # There will be a IterableDataset variant too, TBD
        ds = ImageDataset(root, reader=name, split=split, **kwargs)
    elif name.startswith('tfds/'):
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            is_training=is_training,
            download=download,
            batch_size=batch_size,
            repeats=repeats,
            seed=seed,
            **kwargs
        )
    elif name.startswith('wds/'):
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            is_training=is_training,
            batch_size=batch_size,
            repeats=repeats,
            seed=seed,
            **kwargs
        )
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        if search_split and os.path.isdir(root):
            # look for split specific sub-folder in root
            root = _search_split(root, split)
        #img_mode, corrupted
        if enable_aux:
            # print(f'######################  Enable AUX <= {enable_aux}  ######################')
            if switch_aux < 0:
                ds = MCImageDataset(root, reader=name, class_map=class_map, load_bytes=load_bytes, img_mode=img_mode, corrupted=corrupted, post_dwt=post_dwt, dataset_alias=dataset_alias, switch=sdm,**kwargs)
            else: 
                ds = SwitchmageDataset(root, reader=name, class_map=class_map, load_bytes=load_bytes, img_mode=img_mode, corrupted=corrupted, post_dwt=post_dwt, dataset_alias=dataset_alias, transform_sw=switch_aux, trasnform_k=0, **kwargs)
        else:
            # print(f'######################  Enable AUX <= {enable_aux}  ######################')
            ds = ImageDataset(root, reader=name, class_map=class_map, load_bytes=load_bytes, img_mode=img_mode, corrupted=corrupted, post_dwt=post_dwt, dataset_alias=dataset_alias, **kwargs)
    return ds
