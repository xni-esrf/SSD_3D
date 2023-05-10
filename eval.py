import numpy as np
import torch
import json
import pickle
import os
import sys
import tifffile
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torchio as tio

import datasets 
from models.unet import UNet

def parse_arguments():
    """
     Parses the users command line input
     Returns: NameSpace object containing all needed user inputs
     """
    parse = ArgumentParser(description="Load a model trained with Noise2Iverse, and use it to denoise a volume")
    parse.add_argument('dataset_name', help="Name of the processed dataset. Information about the processed dataset must be set in the dataset.py file")
    parse.add_argument('checkpoint_path', help='The path of the checkpoint to be loaded')
    parse.add_argument('output_dir', help='The directory where the denoised volume is saved')
    parse.add_argument('--extraction_stride_sizes', nargs=3, type=int, default=[50,50,50], help='Specify the stride used to extract successive patches for each dimension. If zeros are given, extracted patches do not overlap.')
    parse.add_argument('--print_orthogonal', action='store_const', default=1, const=3, help='If set, denoised volumes is saved three times. Saved images of each occurence are respectively oriented in the xy, xz, and yz planes. If not set, only the xy plane is considered')
    parse.add_argument('--batch_size', default=32, type=int, help='')
    parse.add_argument('--projection_set', type=str, default="all_projections", help='Specify which recontruction should be loaded. This reconstruction is defined by the projection set used to get it')
    return parse.parse_args()

def load_model(checkpoint_path, cuda_devices):

    model = UNet(
    in_channels= 1,
    out_channels= 1,
    n_blocks= 4,
    start_filts= 40,
    up_mode= 'transpose',
    merge_mode= 'concat',
    planar_blocks= (),
    attention= False,
    activation= 'relu',
    normalization= 'layer',
    full_norm= True,
    dim= 3,
    conv_mode= 'same')

    # Load weights
    state = torch.load(checkpoint_path)
    model.to(cuda_devices[0])
    model = torch.nn.DataParallel(model, cuda_devices)
    model.load_state_dict(state["state_dict"])
    
    # Put model in evaluation mode
    model.eval()
    return model

def save_output(final_vol, print_directions, output_dir):
    for i in range(print_directions):
        Path(output_dir / f"{i}").mkdir(exist_ok=True)
        for j in range(final_vol.shape[0]):
            if i == 0:
                img_np = final_vol[j, :, :]
            if i == 1:
                img_np = final_vol[:, j, :]
            if i == 2:
                img_np = final_vol[:, :, j]
            img_dir = str(output_dir / f"{i}/output_{j:05d}.tif")
            tifffile.imsave(img_dir, img_np)


#Device configuration parameters
cuda_devices = [i for i in range(torch.cuda.device_count())]
cudabase_name = "cuda:" + str(cuda_devices[0])
cudabase = torch.device(cudabase_name)

params = parse_arguments()

# Load the hyper-parameters used to train the loaded model
network_param_file = open(str(Path(params.checkpoint_path).parent/ "params.json"), 'r')
network_params = json.load(network_param_file)
test_patch_size = network_params['train_patch_size']

# Create the directory where denoised slices are saved
output_dir = Path(params.output_dir)
output_dir.mkdir(exist_ok=True)

model = load_model(Path(params.checkpoint_path), cuda_devices)

with torch.no_grad():
    # Build data loading pipeline
    aggregator, loader = datasets.get_test_aggregator_loader(params.dataset_name, test_patch_size, params.extraction_stride_sizes ,params.batch_size, projection_set=params.projection_set)
    # Inference loop
    for batch in tqdm(loader):
        raw_patch, location = batch["test_volume"][tio.DATA],  batch[tio.LOCATION]
        raw_patch.to(cudabase)

        pred_patch = model(raw_patch)
        aggregator.add_batch(pred_patch.detach().cpu(), location)    
    pred_volume = aggregator.get_output_tensor()

pred_volume = np.asarray(pred_volume)
pred_volume = np.squeeze(pred_volume)

# Save denoised volume
save_output(pred_volume, params.print_orthogonal, output_dir)












