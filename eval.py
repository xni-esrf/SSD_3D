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

def bit_quantization(array, num_bits):
    print("quantization")
    max_value = np.max(array)
    min_value = np.min(array)
    step_size = (max_value - min_value) / (2 ** num_bits - 1)
    quantized_array = np.round((array - min_value) / step_size) * step_size + min_value
    return quantized_array

def parse_arguments():
    """
     Parses the users command line input
     Returns: NameSpace object containing all needed user inputs
     """
    parse = ArgumentParser(description="Denoise a dataset using a given Noise2Inverse model")
    parse.add_argument('dataset_name', help="Dataset name")
    parse.add_argument('weights_path', help='The torch file containing the trained weights to use for denoising')
    parse.add_argument('output_dir', help='The directory that the final denoised images will be output to ')
    parse.add_argument('--overlap_patch_size', nargs=3, type=int, default=[100,100,100], help='overlapping area to crop around each predicted patch')
    parse.add_argument('--overlap_mode', type=str, default="hann", help='')
    parse.add_argument('--print_orthogonal', action='store_const', default=1, const=3, help='Slice the final denoised volume in the xy, yz, and xz planes. Slices only in the xy if off')
    parse.add_argument('--batch_size', default=32, type=int, help='')
    parse.add_argument('--projection_set', type=str, default="all_projections", help='')
    return parse.parse_args()

def load_model(weights_path, cuda_devices):

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
    state = torch.load(weights_path)
    model.to(cuda_devices[0])
    model = torch.nn.DataParallel(model, cuda_devices)
    model.load_state_dict(state["state_dict"])
    
    # Put model in evaluation mode
    model.eval()
    return model

def save_output(final_vol, print_directions, output_dir):
    """
    Print out the final denoised volume as a series of 2D tiff file slices
    Args:
        final_vol: The final denoised volume as a numpy array
        print_directions: The number of directions to output slices from
        output_dir: The directory where output images will be saved

    Returns:

    """
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
# Load in the parameters of the trained network
network_param_file = open(str(Path(params.weights_path).parent/ "params.json"), 'r')
network_params = json.load(network_param_file)
test_patch_size = network_params['train_patch_size']

# Make the folders where denoised slices are saved
if params.output_dir[-1] == '/':
	params.output_dir = params.output_dir[0:-1]
if os.path.exists(params.output_dir[0:-len(params.output_dir.split('/')[-1])]) == False:
	os.mkdir(params.output_dir[0:-len(params.output_dir.split('/')[-1])])
if os.path.exists(params.output_dir)  == False:
	os.mkdir(params.output_dir)

# Load in model to use for denoising
model = load_model(Path(params.weights_path), cuda_devices)

# Prepare the dataset to be denoised
with torch.no_grad():
    aggregator, loader = datasets.get_test_aggregator_loader(params.dataset_name, test_patch_size, params.overlap_patch_size,  params.overlap_mode ,params.batch_size, projection_set=params.projection_set)
    for batch in tqdm(loader):
        raw_patch, location = batch["test_volume"][tio.DATA],  batch[tio.LOCATION]
        raw_patch.to(cudabase)
        pred_patch = model(raw_patch)

        aggregator.add_batch(pred_patch.detach().cpu(), location)
    pred_volume = aggregator.get_output_tensor()

pred_volume = np.asarray(pred_volume)
pred_volume = np.squeeze(pred_volume)

#pred_volume = bit_quantization(pred_volume, 8)

# Save denoised volume
save_output(pred_volume, params.print_orthogonal, Path(params.output_dir))












