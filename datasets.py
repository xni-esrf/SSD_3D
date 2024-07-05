import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import json

import torchio as tio


# Get (1) path to dataset volumes, (2) volume shape, (3) crop shape if any, (4) crop position if any
# These info must be specified in the metadata folder
def get_dataset_info(dataset_name):
    with open("metadata/" + dataset_name + ".json", 'r') as dataset_info_file:
        dataset_info = json.load(dataset_info_file)
    return dataset_info

# Pick one of the 24 rotational symmetries of a cube. Give this orientation to the provided tensors. A horizontal flip is then randomly applied with a 0.5 probability.
# Note that the same transformation is applied to all given tensors.
def geom_transform(tensors_to_transform):

    cur_direction = torch.randint(0,3,[1])
    cur_channel_flip = torch.randint(0,2,[1])
    cur_nb_rot = torch.randint(0,4,[1])
    cur_h_flip = torch.randint(0,2,[1])

    for i in range(len(tensors_to_transform)):
        # Select one of 6 cube face as an input face
        tensors_to_transform[i] = torch.squeeze(tensors_to_transform[i])
        if cur_direction.item() == 1:
            tensors_to_transform[i]=torch.swapaxes(tensors_to_transform[i],0,1)
        elif cur_direction.item() == 2:
            tensors_to_transform[i]=torch.swapaxes(tensors_to_transform[i],0,2)
        if cur_channel_flip.item() == 1:
            tensors_to_transform[i]=torch.flip(tensors_to_transform[i], [0])

        # Rotate around the axis orthoghonal to input face selected above
        tensors_to_transform[i]=torch.rot90(tensors_to_transform[i], cur_nb_rot.item(), [1,2])

        # Randomly horizontally flip
        if cur_h_flip.item() == 1:
            tensors_to_transform[i]=torchvision.transforms.functional.hflip(tensors_to_transform[i])
        tensors_to_transform[i] = torch.unsqueeze(tensors_to_transform[i],0)
    return tensors_to_transform


def get_train_patch_loader(dataset_name, training_patch_size, batch_size, nb_patch_per_epoch, normalization):

    # Load the dataset meta-data
    dataset_info = get_dataset_info(dataset_name)
    split1_volume_path = dataset_info["split1_volume_path"]
    split2_volume_path = dataset_info["split2_volume_path"]
    volume_shape = dataset_info["volume_shape"]         
    offset_crop = dataset_info["offset_crop"]
    crop_subvolume_shape = dataset_info["crop_subvolume_shape"]

    # Load the two training volumes as numpy arrays.
    split1_volume = np.fromfile(split1_volume_path, dtype=np.float32).reshape(volume_shape)
    split2_volume = np.fromfile(split2_volume_path, dtype=np.float32).reshape(volume_shape)

    # If specified, crop the training volumes
    if crop_subvolume_shape:
        split1_volume = split1_volume[offset_crop[0]:crop_subvolume_shape[0]+offset_crop[0],offset_crop[1]:crop_subvolume_shape[1]+offset_crop[1],offset_crop[2]:crop_subvolume_shape[2]+offset_crop[2]]
        split2_volume = split2_volume[offset_crop[0]:crop_subvolume_shape[0]+offset_crop[0],offset_crop[1]:crop_subvolume_shape[1]+offset_crop[1],offset_crop[2]:crop_subvolume_shape[2]+offset_crop[2]]

    # If provided shape is 3D, add an extra dimension to form the pytorch channel dimension
    if len(volume_shape) == 3:
        split1_volume = np.expand_dims(split1_volume,0)
        split2_volume = np.expand_dims(split2_volume,0)
    
    if normalization==True:
        print("Dataset normalization in progress...")
        mean_volumes = (split1_volume.mean() + split2_volume.mean())/2
        split1_volume = split1_volume-mean_volumes
        split2_volume = split2_volume-mean_volumes

        std_volumes = (split1_volume.std() + split2_volume.std())/2
        split1_volume = split1_volume/std_volumes
        split2_volume = split2_volume/std_volumes

    N2I_subject = tio.Subject(split1_volume=tio.ScalarImage(tensor=split1_volume), split2_volume=tio.ScalarImage(tensor=split2_volume))
    N2I_dataset = tio.data.SubjectsDataset([N2I_subject])

    patch_sampler = tio.data.UniformSampler(training_patch_size)
    N2I_queue = tio.data.Queue(N2I_dataset, max_length=batch_size*nb_patch_per_epoch, samples_per_volume=nb_patch_per_epoch, sampler=patch_sampler, num_workers=0)


    # With queue, num_worker in dataloader must be equals to zero as said in the tio queue documentation
    patch_loader = DataLoader(N2I_queue, batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)

    return patch_loader

def get_list_train_patch_loader(list_dataset_names, training_patch_size, batch_size, nb_patch_per_epoch):
    
    assert batch_size%(len(list_dataset_names)*2) ==0
    assert nb_patch_per_epoch%len(list_dataset_names) ==0

    list_loaders = []
    for name in list_dataset_names:
        cur_loader = get_train_patch_loader(name, training_patch_size, batch_size//len(list_dataset_names), nb_patch_per_epoch//len(list_dataset_names))
        list_loaders.append(cur_loader)
    return list_loaders

def get_test_aggregator_loader(dataset_name, test_patch_size, patch_overlap, batch_size, projection_set, normalization):

    # Load the dataset meta-data
    dataset_info = get_dataset_info(dataset_name)
    volume_shape = dataset_info["volume_shape"]
    offset_crop = dataset_info["offset_crop"]
    crop_subvolume_shape = dataset_info["crop_subvolume_shape"]
    # Pick the dataset that have been reconstructed using the specified projection set.
    if projection_set == "all_projections":
        test_volume_path = dataset_info["test_volume_path"]
    elif projection_set == "split1_projections":
        test_volume_path = dataset_info["split1_volume_path"]
    elif projection_set == "split2_projections":
        test_volume_path = dataset_info["split2_volume_path"]
    else:
        raise Exception("Projection set name can only be : all_projections, split1_projections or split2_projections")

    # Load volume to be denoise. Then crop if specified
    test_volume = np.fromfile(test_volume_path, dtype=np.float32).reshape(volume_shape)
    if crop_subvolume_shape:
        test_volume = test_volume[offset_crop[0]:crop_subvolume_shape[0]+offset_crop[0],offset_crop[1]:crop_subvolume_shape[1]+offset_crop[1],offset_crop[2]:crop_subvolume_shape[2]+offset_crop[2]]

    # If provided shape is 3D, add an extra dimension to form the pytorch channel dimension
    assert len(volume_shape) == 3
    test_volume = np.expand_dims(test_volume,0)

    if normalization==True:
        print("Dataset normalization in progress...")
        mean_volume = test_volume.mean()
        test_volume = test_volume-mean_volume

        std_volume = test_volume.std()
        test_volume = test_volume/std_volume


    # Build the data loading pipeline
    test_subject = tio.Subject(test_volume=tio.ScalarImage(tensor=test_volume))
    sampler = tio.GridSampler(test_subject, test_patch_size, patch_overlap)
    aggregator = tio.data.GridAggregator(sampler, overlap_mode="hann")
    loader=DataLoader(sampler, batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=0)

    return aggregator, loader



