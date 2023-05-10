import json
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import torchio as tio

import datasets
from models.unet import UNet


def parse_arguments():
    parse = ArgumentParser(description="Train a model with Noise2Inverse, using 3d convolutions")
    parse.add_argument('dataset_name', help='Name of a json file in the metadata folder, which is used to load information about the processed dataset')
    parse.add_argument('checkpoint_dir', help='Path of the directory where checkpoints are stored')
    parse.add_argument('loss_dir', help='Path of the directory where the training loss curve is stored')
    parse.add_argument('--loaded_checkpoint_path', default=None, help="If set, load the checkpoint located at the provided path")
    parse.add_argument('--train_patch_size', default=[100,100,100], nargs=3, type=int, help='Training patch size')
    parse.add_argument('--nb_patch_per_epoch', default=18000, type=int, help="Define the number of patches to process in one epoch. It basically sets the duration of one epoch") 
    parse.add_argument('--nb_train_epoch', default=100, type=int, help="The number of training epochs")
    parse.add_argument('--save_interval', default=5, type=int, help='The number of epochs to run between each checkpoint saving')
    parse.add_argument('--batch_size', default=32, type=int, help="The number of patch per batch")
    parse.add_argument('--lr', default=0.001, type=float, help="The learning rate")
    parse.add_argument('--weight_decay', default=0.0001, type=float, help="The weight decay coefficient")
    return parse.parse_args()


def make_model(cuda_devices):
	
    model = UNet(
    in_channels= 1,
    out_channels= 1,
    n_blocks= 4,
    start_filts=40,
    up_mode= 'transpose',
    merge_mode= 'concat',
    planar_blocks= (),
    attention= False,
    activation= 'relu',
    normalization= 'layer',
    full_norm= True,
    dim= 3,
    conv_mode= 'same')

    model.to(torch.device("cuda:0"))
    model = torch.nn.DataParallel(model,cuda_devices)

    return model

def save_model(model, optimizer, epoch, save_path):
    state = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, save_path)

def train_model(dl, model, loss_func, optimizer, save_interval, checkpoint_dir, loaded_checkpoint_path, nb_train_epoch, device):

    start_epoch_nb = 0

    # Load in old weights if required
    if loaded_checkpoint_path:
        print("Loading weights...")
        state = torch.load(loaded_checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch_nb = state['epoch']+1
        print(start_epoch_nb)

    # training loop
    loss_values = []
    for epoch in range(start_epoch_nb, nb_train_epoch):
        epoch_loss = 0
        for batch in tqdm(dl):

            # Constitute input and target with volumes extrated from split1_volume and split2_volume
            input = torch.concat([batch["split1_volume"][tio.DATA][0:dl.batch_size//2], batch["split2_volume"][tio.DATA][dl.batch_size//2:]], 0)
            target = torch.concat([batch["split2_volume"][tio.DATA][0:dl.batch_size//2], batch["split1_volume"][tio.DATA][dl.batch_size//2:]], 0)

            # For each input-target training couple, pick a random geometric transformation, and apply it to both input and target
            for i in range(input.shape[0]):
                input[i], target[i] = datasets.geom_transform([input[i], target[i]])
            input, target = input.to(device), target.to(device)

            # Shuffle element in batch
            random_perm = torch.randperm(dl.batch_size)
            input, target = input[random_perm], target[random_perm]
            
            # Proceed to a training step
            optimizer.zero_grad()
            pred = model(input)
            loss_val = loss_func(pred,target)
            loss_val.backward()
            optimizer.step()

            epoch_loss += loss_val/len(dl)
        print("Mean loss value of the epoch : {}".format(epoch_loss))

        # Save network if the interval has been reached, or if training ends
        if epoch % save_interval == 0 or epoch==nb_train_epoch-1:
            print("saving interval reached (epoch n°{}), saving checkpoint...".format(epoch))
            save_model(model, optimizer, epoch, checkpoint_dir / f"weights_epoch_{epoch}.torch")
        # Recording current epoch loss value
        loss_values.append(epoch_loss.detach().cpu())

    return loss_values


def save_train_loss(loss_values, loss_dir):

    map_epoch_loss = dict(zip(list(range(len(loss_values))), loss_values))

    best_epoch = list(dict(sorted(map_epoch_loss.items(), key=lambda item: float(item[1]), reverse=True)))[-1]

    plt.plot(loss_values)
    plt.xticks(list(range(0,len(loss_values),5)))
    plt.ylabel("Train_Loss")
    plt.xlabel("Epoch, (minimum reached at epoch = {}, corresponding_value : {})".format(best_epoch,map_epoch_loss[best_epoch]))
    plt.title(loss_dir.name)

    plt.savefig(loss_dir)


params = parse_arguments()

# Create output directories if they do not exist
checkpoint_dir = Path(params.checkpoint_dir)
loss_dir = Path(params.loss_dir)
checkpoint_dir.mkdir(exist_ok=True)
loss_dir.mkdir(exist_ok=True)

#List available GPUs
cuda_devices = [i for i in range(torch.cuda.device_count())]

# Save the training parameters
with open(checkpoint_dir / "params.json", 'w') as par_file:
    json.dump(dict(vars(params)), par_file)

# Initialize the model to be trained
model = make_model(cuda_devices)

# Training patch size must be divisible by the number of blocks of the trained 3D U-Net
assert params.train_patch_size[-3]%model.module.n_blocks == 0 and params.train_patch_size[-1]%model.module.n_blocks == 0 and params.train_patch_size[-2]%model.module.n_blocks == 0

# Get the data loading pipeline
train_loader = datasets.get_train_patch_loader(params.dataset_name, params.train_patch_size, params.batch_size, params.nb_patch_per_epoch)

# Train model
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=params.weight_decay, lr=params.lr)
loss_values = train_model(train_loader, model, loss_func, optimizer, params.save_interval, checkpoint_dir, params.loaded_checkpoint_path, params.nb_train_epoch, torch.device("cuda:0"))

save_train_loss(loss_values, loss_dir)
