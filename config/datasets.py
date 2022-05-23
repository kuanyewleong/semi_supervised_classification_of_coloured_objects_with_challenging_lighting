import torchvision.transforms as transforms
from .augmentations import RandAugment
from .utils import export
import os


@export
def star():
    channel_stats = dict(mean=[0.0396, 0.0273, 0.0230],
                         std=[0.0917, 0.0671, 0.0535])
    
    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((20,200), padding=4,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((20,200), padding=4,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((20, 200)), # added by Leong, to solve the issue of tensor size mismatch
        transforms.Normalize(**channel_stats)
    ])


    myhost = os.uname()[1]
    data_dir = '/../../media/16TBHDD/leong/dataset/star/semi_supervised'

    print("Using theStar from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 15
    }


@export
def r304():
    channel_stats = dict(mean=[0.0536, 0.0469, 0.0472],
                         std=[0.1033, 0.0853, 0.0898])
    
    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((20,200), padding=4,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((20,200), padding=4,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((20, 200)), # added by Leong, to solve the issue of tensor size mismatch
        transforms.Normalize(**channel_stats)
    ])


    myhost = os.uname()[1]
    data_dir = '/../../media/16TBHDD/leong/dataset/star/semi_supervised/Room304'
    
    print("Using 304 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 15
    }


@export
def area3():
    channel_stats = dict(mean=[0.0359, 0.0248, 0.0214],
                         std=[0.0648, 0.0414, 0.0293])
    
    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((20,200), padding=4,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((20,200), padding=4,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((20, 200)), # added by Leong, to solve the issue of tensor size mismatch
        transforms.Normalize(**channel_stats)
    ])


    myhost = os.uname()[1]
    data_dir = '/../../media/16TBHDD/leong/dataset/star/semi_supervised/Area3'
    
    print("Using Area3 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 15
    }