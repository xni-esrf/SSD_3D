import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import torchio as tio



def get_dataset_info(dataset_name):
    
    dataset_into = {}

    if dataset_name[0:7] == "CB_30nm":
        dataset_into["test_volume_path"] = "/data/id16a/inhouse4/staff/ap/liam/neural-super-res/data/n2i_reduced_angles/volFilesCBxs_lobV/CBxs_lobV_bottomp100um_30nm_rectwopassdb9_crop.vol"
        dataset_into["split1_volume_path"] = "/data/id16a/inhouse4/staff/ap/liam/neural-super-res/data/n2i_reduced_angles/volFilesCBxs_lobV/CBxs_lobV_bottomp100um_30nm_rectwopassdb9_split1.vol"
        dataset_into["split2_volume_path"] = "/data/id16a/inhouse4/staff/ap/liam/neural-super-res/data/n2i_reduced_angles/volFilesCBxs_lobV/CBxs_lobV_bottomp100um_30nm_rectwopassdb9_split2.vol"
        dataset_into["volume_shape"] = [1500,1500,1500]
        if dataset_name == "CB_30nm":
            dataset_into["offset_crop"]=None
            dataset_into["crop_subvolume_shape"]=None
        elif dataset_name == "CB_30nm_1000x1000":
            dataset_into["offset_crop"]=[500,500,500]
            dataset_into["crop_subvolume_shape"]=[1000,1000,1000]
        elif dataset_name == "CB_30nm_500x500":
            dataset_into["offset_crop"]=[500,500,500]
            dataset_into["crop_subvolume_shape"]=[500,500,500]
        elif dataset_name == "CB_30nm_500x500_test":
            dataset_into["offset_crop"]=[0,1000,1000]
            dataset_into["crop_subvolume_shape"]=[500,500,500]

    elif dataset_name[0:7]=="OB_25nm":
        dataset_into["test_volume_path"] = "/data/projects/xni/ls2892/volfloat/c353_epl_70um_ppy_md_025nm_rec_db15_.vol"
        dataset_into["split1_volume_path"] = "/data/projects/xni/ls2892/volfloat/c353_epl_70um_ppy_md_025nm_rec_db15_split0.vol"
        dataset_into["split2_volume_path"] = "/data/projects/xni/ls2892/volfloat/c353_epl_70um_ppy_md_025nm_rec_db15_split1.vol"
        dataset_into["volume_shape"] = [2048,2400,2400]
        if dataset_name == "OB_25nm":
            dataset_into["offset_crop"]=[274,450,450]
            dataset_into["crop_subvolume_shape"]=[1500,1500,1500]
        elif dataset_name == "OB_25nm_1000x1000":
            dataset_into["offset_crop"]=[774,950,950]
            dataset_into["crop_subvolume_shape"]=[1000,1000,1000]
        elif dataset_name == "OB_25nm_500x500":
            dataset_into["offset_crop"]=[774,950,950]
            dataset_into["crop_subvolume_shape"]=[500,500,500]
        elif dataset_name == "OB_25nm_500x500_test":
            dataset_into["offset_crop"]=[274,450,1450]
            dataset_into["crop_subvolume_shape"]=[500,500,500]

    elif dataset_name[0:27]=="distsplit_OB_25nm_1000x1000":
        dataset_into["test_volume_path"] = "/data/visitor/ls2892/id16a/c353_epl_70um_ppy/.volfloat_nobackup/c353_epl_70um_ppy_md_025nm_rec_.vol"
        dataset_into["split1_volume_path"] = "/data/visitor/ls2892/id16a/c353_epl_70um_ppy/.volfloat_nobackup/c353_epl_70um_ppy_md_025nm_rec_split13_.vol"
        dataset_into["split2_volume_path"] = "/data/visitor/ls2892/id16a/c353_epl_70um_ppy/.volfloat_nobackup/c353_epl_70um_ppy_md_025nm_rec_split24_.vol"
        dataset_into["volume_shape"] = [2048,2048,2048]
        dataset_into["offset_crop"]=[524,524,524]
        dataset_into["crop_subvolume_shape"]=[1000,1000,1000]

    elif dataset_name[0:27]=="distsplit_OB_25nm_1500x1500":
        dataset_into["test_volume_path"] = "/data/visitor/ls2892/id16a/c353_epl_70um_ppy/.volfloat_nobackup/c353_epl_70um_ppy_md_025nm_rec_.vol"
        dataset_into["split1_volume_path"] = "/data/visitor/ls2892/id16a/c353_epl_70um_ppy/.volfloat_nobackup/c353_epl_70um_ppy_md_025nm_rec_split13_.vol"
        dataset_into["split2_volume_path"] = "/data/visitor/ls2892/id16a/c353_epl_70um_ppy/.volfloat_nobackup/c353_epl_70um_ppy_md_025nm_rec_split24_.vol"
        dataset_into["volume_shape"] = [2048,2048,2048]
        dataset_into["offset_crop"]=[274,274,274]
        dataset_into["crop_subvolume_shape"]=[1500,1500,1500]

    elif dataset_name[0:7]=="WB_50nm":
        dataset_into["test_volume_path"] = "/data/projects/xni/ls3033/id16a/WB3/volfloat/WB3_hinge4_050nm_rec27db_crop.vol"
        dataset_into["split1_volume_path"] = "/data/projects/xni/ls3033/id16a/WB3/volfloat/WB3_hinge4_050nm_rec27db_split1.vol"
        dataset_into["split2_volume_path"] = "/data/projects/xni/ls3033/id16a/WB3/volfloat/WB3_hinge4_050nm_rec27db_split2.vol"
        dataset_into["volume_shape"] = [1024,512,512]
        dataset_into["offset_crop"]=[100,6,6]
        dataset_into["crop_subvolume_shape"] = [500,500,500]

    elif dataset_name[0:9]== "V357_18nm":
        dataset_into["test_volume_path"] = "/data/id16a/inhouse4/staff/ap/liam/neural-super-res/data/temp/v357_55um_3K_018nm_rec_app1_1500x1500/v357_55um_3K_018nm_rec_app1_1500x1500_crop.vol"
        dataset_into["split1_volume_path"] = "/data/id16a/inhouse4/staff/ap/liam/neural-super-res/data/temp/v357_55um_3K_018nm_rec_app1_1500x1500/v357_55um_3K_018nm_rec_app1_1500x1500_split0.vol"
        dataset_into["split2_volume_path"] = "/data/id16a/inhouse4/staff/ap/liam/neural-super-res/data/temp/v357_55um_3K_018nm_rec_app1_1500x1500/v357_55um_3K_018nm_rec_app1_1500x1500_split1.vol"
        dataset_into["volume_shape"] = [1500,1500,1500]
        dataset_into["offset_crop"]=[0,0,0]
        dataset_into["crop_subvolume_shape"] = None

    elif dataset_name=="ring_test_1":
        dataset_into["split1_volume_path"] = "/data/id16a/inhouse4/staff/dk/cont_denoise_test/aligned_volumes/pos1.tif"
        dataset_into["split2_volume_path"] = "/data/id16a/inhouse4/staff/dk/cont_denoise_test/aligned_volumes/pos2.tif"
        dataset_into["volume_shape"] = [1000,1000,1000]
        dataset_into["offset_crop"]=[0,0,0]
        dataset_into["crop_subvolume_shape"] = None

    elif dataset_name=="ring_test_2":
        dataset_into["split1_volume_path"] = "/data/projects/xni/staff/alaugros/noise2inverse/datasets/ring_test_2/split1.npy"
        dataset_into["split2_volume_path"] = "/data/projects/xni/staff/alaugros/noise2inverse/datasets/ring_test_2/split2.npy"
        dataset_into["volume_shape"] = [1500,1500,1500]
        dataset_into["offset_crop"]=[0,0,0]
        dataset_into["crop_subvolume_shape"] = None

    elif dataset_name=="X2O2_30nm":
        dataset_into["split1_volume_path"] = "/data/projects/xni/staff/alaugros/noise2inverse/datasets/X2O2_WM_cont_030nm_a_registered/split1.npy"
        dataset_into["split2_volume_path"] = "/data/projects/xni/staff/alaugros/noise2inverse/datasets/X2O2_WM_cont_030nm_a_registered/split2.npy"
        dataset_into["volume_shape"] = [1500,1500,1500]
        dataset_into["offset_crop"]=[0,0,0]
        dataset_into["crop_subvolume_shape"] = None

    else:
        raise Exception("No information about provided dataset name") 

    return dataset_into


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


def get_train_patch_loader(dataset_name, training_patch_size, batch_size, nb_patch_per_epoch):

    dataset_info = get_dataset_info(dataset_name)

    split1_volume_path = dataset_info["split1_volume_path"]
    split2_volume_path = dataset_info["split2_volume_path"]
    volume_shape = dataset_info["volume_shape"]         
    offset_crop = dataset_info["offset_crop"]
    crop_subvolume_shape = dataset_info["crop_subvolume_shape"]

    # build numpy arrays from provided path
    split1_volume = np.fromfile(split1_volume_path, dtype=np.float32).reshape(volume_shape)
    split2_volume = np.fromfile(split2_volume_path, dtype=np.float32).reshape(volume_shape)

    # If specified, crop a subvolume
    if crop_subvolume_shape:
        split1_volume = split1_volume[offset_crop[0]:crop_subvolume_shape[0]+offset_crop[0],offset_crop[1]:crop_subvolume_shape[1]+offset_crop[1],offset_crop[2]:crop_subvolume_shape[2]+offset_crop[2]]
        split2_volume = split2_volume[offset_crop[0]:crop_subvolume_shape[0]+offset_crop[0],offset_crop[1]:crop_subvolume_shape[1]+offset_crop[1],offset_crop[2]:crop_subvolume_shape[2]+offset_crop[2]]

    # If provided shape is 3D, add en extra diemnsion for pytorch channels
    if len(volume_shape) == 3:
        split1_volume = np.expand_dims(split1_volume,0)
        split2_volume = np.expand_dims(split2_volume,0)

    N2I_subject = tio.Subject(split1_volume=tio.ScalarImage(tensor=split1_volume), split2_volume=tio.ScalarImage(tensor=split2_volume))
    N2I_dataset = tio.data.SubjectsDataset([N2I_subject])

    patch_sampler = tio.data.UniformSampler(training_patch_size)
    # TODO, investigate why trainings get slower when num_worker > 0
    N2I_queue = tio.data.Queue(N2I_dataset, max_length=batch_size*nb_patch_per_epoch, samples_per_volume=nb_patch_per_epoch, sampler=patch_sampler, num_workers=0)


    # While using uniform patch loader, shuffle is not relevant. With queue, num_worker in dataloader must be equals to zero as said in the tio queue documentation
    patch_loader = DataLoader(N2I_queue, batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)

    return patch_loader


def get_test_aggregator_loader(dataset_name, test_patch_size, patch_overlap, overlap_mode, batch_size, projection_set="all_projections"):

    dataset_info = get_dataset_info(dataset_name)

    if projection_set == "all_projections":
        test_volume_path = dataset_info["test_volume_path"]
    elif projection_set == "split1_projections":
        test_volume_path = dataset_info["split1_volume_path"]
    elif projection_set == "split2_projections":
        test_volume_path = dataset_info["split2_volume_path"]

    volume_shape = dataset_info["volume_shape"]
    offset_crop = dataset_info["offset_crop"]
    crop_subvolume_shape = dataset_info["crop_subvolume_shape"]

    # build numpy arrays from provided path
    test_volume = np.fromfile(test_volume_path, dtype=np.float32).reshape(volume_shape)

    # If specified, crop a subvolume
    if crop_subvolume_shape:
        test_volume = test_volume[offset_crop[0]:crop_subvolume_shape[0]+offset_crop[0],offset_crop[1]:crop_subvolume_shape[1]+offset_crop[1],offset_crop[2]:crop_subvolume_shape[2]+offset_crop[2]]

    # Provided shape should be 3D, add en extra dimension for pytorch channels
    assert len(volume_shape) == 3
    test_volume = np.expand_dims(test_volume,0)

    test_subject = tio.Subject(test_volume=tio.ScalarImage(tensor=test_volume))

    sampler = tio.GridSampler(test_subject, test_patch_size, patch_overlap)

    aggregator = tio.data.GridAggregator(sampler, overlap_mode=overlap_mode)

    loader=DataLoader(sampler, batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=0)

    return aggregator, loader



