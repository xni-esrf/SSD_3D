# XNI Noise2Inverse Implementation

Implementation of the Noise2Inverse algorithm adaptated to 3D data.
This code corresponds to the "3d implementation" used in the paper entitled "Self-supervised image restoration in coherent X-ray neuronal microscopy".


## Installation

The commands provided below, work in a conda environment that can be created with: 

```
conda env create -f environment.yml
```

To activate the created environment:

```
conda activate noise2inverse_3d_xni
```

To test this code, you can download a volume imaging a fly neuropile here : https://drive.google.com/file/d/1tEbyaTwU0S8uQ21TYoluTgclkJflORvl/view?usp=sharing
The commands below assume that the downloaded archive is extracted in the 'volumes' folder

## Dataset Metadata

Before launching a training on a dataset, information about this dataset must be provided in a json file stored in the metadata folder.
To give an example, the metadata of the provided fly neuropil volume can be found in the metadata folder.
For any datasets, metadata that must be provided are: path to the volume reconstructed using all projections, the paths of the volumes reconsutructed using a projection subset, volume size, position of an potential crop, Size of the potential crop.


## Demonstration Commands

Launch a Noise2Inverse training on the provided volume by running:
```
python train.py NP_50nm results/checkpoints/test results/train_losses/test --nb_train_epoch 30
```

Denoise the provided volume using the model trained above by running:
```
python eval.py NP_50nm results/checkpoints/test/weights_epoch_10.torch results/denoised_volumes/test/epoch10 --print_orthogonal --nb_bit_quant 8
```

Effective resolution can be obtained in two steps (1) denoise the two volumes that have been reconstructed using a projection subset, (2) Compute a fourier shell correlation curve using the denoised volumes. This can be done by running:

```
python eval.py NP_50nm results/checkpoints/test/weights_epoch_10.torch results/denoised_volumes/test/epoch10_split1_8bit --projection_set split1_projections --nb_bit_quant 8
python eval.py NP_50nm results/checkpoints/test/weights_epoch_10.torch results/denoised_volumes/test/epoch10_split2_8bit --projection_set split2_projections --nb_bit_quant 8
python fsc_analysis_multi_plots.py results/denoised_volumes/test/epoch10_split1_8bit/0 results/denoised_volumes/test/epoch10_split2_8bit/0 results/FSC_curves/test epoch10_8bit
```


# Citation