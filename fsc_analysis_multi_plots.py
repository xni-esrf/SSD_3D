import os
from pathlib import Path
import re
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from FSC import fsc_corrct_tools
from tools.quantization import bit_quantization

def parse_arguments():
    parse = ArgumentParser(description="Computes the Fourier Shell Correlation between volume1 and volume2, and computes the threshold funcion T of 1/2 bit.")
    parse.add_argument('vol1_path', help='The directory containing the split0 volume')
    parse.add_argument('vol2_path', help='The directory containing the split1 volume')
    parse.add_argument('output_dir', help="The path of the directory where the plot is saved")
    parse.add_argument('file_name', default="", help="The name of the saved file")
    parse.add_argument('--nb_bit_quant', default=None, type=int, help='Quantize value of input arrays before computing FSC. The number of quantization levels equals 2^num_bits')

    return parse.parse_args()


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


if __name__ == '__main__':

    parser = parse_arguments()
    vol1_path = parser.vol1_path
    vol2_path = parser.vol2_path
    output_dir = parser.output_dir
    file_name = parser.file_name
    nb_bit_quant = parser.nb_bit_quant

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Reading tiff files to build input volumes...")
    vol1_fnames = [str(vol1_path)+'/'+f for f in os.listdir(vol1_path) if os.path.isfile(os.path.join(vol1_path, f))]
    vol1_fnames.sort(key=natural_keys)
    vol2_fnames = [str(vol2_path)+'/'+f for f in os.listdir(vol2_path) if os.path.isfile(os.path.join(vol2_path, f))]
    vol2_fnames.sort(key=natural_keys)

    vol1 = [tifffile.imread(fname) for fname in vol1_fnames]
    vol2 = [tifffile.imread(fname) for fname in vol2_fnames]
    vol1 = np.array(vol1)
    vol2 = np.array(vol2)

    if nb_bit_quant:
        print("Quantizing pixel values...")
        vol1 = bit_quantization(vol1,nb_bit_quant)
        vol2 = bit_quantization(vol2,nb_bit_quant)


    print("Computing FSC...")
    fig, ax, res = fsc_corrct_tools.plot_frcs([(vol1,vol2)],(file_name), taper_ratio=0.05, smooth=20)
    plt.savefig(os.path.join(output_dir,file_name))
    plt.clf()

