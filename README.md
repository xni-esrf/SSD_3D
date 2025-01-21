Code associated with the paper entitled "Self-supervised image restoration in coherent X-ray neuronal microscopy".


## Installation

The commands provided below, work in a conda environment that can be created with: 

```
conda env create -f environment.yml
```

To activate the created environment:

```
conda activate n2i_xni_cuda12
```

To test this code, you can download a volume imaging a fly neuropile here : https://drive.google.com/file/d/1tEbyaTwU0S8uQ21TYoluTgclkJflORvl/view?usp=sharing
All commands below assume that the downloaded archive is extracted in the 'volumes' folder

## Dataset Metadata

Before launching a training on a dataset, information about this dataset must be provided in a json file stored in the metadata folder.
For example, the metadata of the provided fly neuropil volume can be found in the metadata folder.
For any datasets, metadata that must be provided are: path to the volume reconstructed using all projections, the paths of the volumes reconsutructed using a projection subset, volume size, position of an potential crop, size of the potential crop.


## Demonstration Commands

Launch a training on the provided volume by running:
```
python train.py NP_50nm results/checkpoints/test results/train_losses/test --nb_train_epoch 30 --normalization
```
Please note that this training requires more than 100GB GPU RAM. Consider reducing patch size of batch size when having less RAM available.

Denoise the provided volume using the model trained above by running:
```
python eval.py NP_50nm results/checkpoints/test/weights_epoch_15.torch results/denoised_volumes/test/epoch15 --print_orthogonal
```

Effective resolution can be obtained in two steps (1) denoise the two training volumes, (2) Compute a fourier shell correlation curve using the denoised volumes. This can be done by running:

```
python eval.py NP_50nm results/checkpoints/test/weights_epoch_15.torch results/denoised_volumes/test/epoch15_split1 --projection_set split1_projections
python eval.py NP_50nm results/checkpoints/test/weights_epoch_15.torch results/denoised_volumes/test/epoch15_split2 --projection_set split2_projections
python fsc_analysis_multi_plots.py results/denoised_volumes/test/epoch15_split1/0 results/denoised_volumes/test/epoch15_split2/0 results/FSC_curves/test epoch15.png
```


# Citation
