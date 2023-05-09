import json
import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import torchvision
import torchio as tio

import datasets
from models.unet import UNet


def parse_arguments():
    parse = ArgumentParser(description="Train a model using the Noise2Inverse procedure")
    parse.add_argument('dataset_name', help='the directory containing the training set of tiff images')
    parse.add_argument('output_dir', help='the directory where the weights files will be stored')
    parse.add_argument('loss_fn', help='the file where the training loss files will be stored (json)')
    parse.add_argument('--weights_path', default=None, help="the weights file to load in before training, default none")
    parse.add_argument('--train_patch_size', default=[100,100,100], nargs=3, type=int, help='Size of the extracted patches')
    parse.add_argument('--nb_patch_per_epoch', default=18000, type=int, help="Default value corresponds to the number of 100x100x100 patches to extract to cover two splits dataset of 500x500 size, with the 48 possible geom transform") 
    parse.add_argument('--loss_func', default="MSE", help="loss function to use, default L2")
    parse.add_argument('--nb_train_epoch', default=100, type=int, help="number of training epochs")
    parse.add_argument('--start_epoch', default=0, type=int, help="if training from old weights, which epoch to start at")
    parse.add_argument('--save_interval', default=5, type=int, help='how many epochs to run in between weights file saving')
    parse.add_argument('--batch_size', default=32, type=int, help="number of patch per batch")
    parse.add_argument('--lr', default=0.001, type=float, help="learning rate")
    parse.add_argument('--weight_decay', default=0.0001, type=float, help="coefficient of the used weight decay regularization")
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

def train_model(dl, model, loss_func, optimizer, save_interval, output_dir, weights_path, start_epoch, nb_train_epoch, device):

    # Load in old weights if required
    if weights_path:
        print("Loading weights...")
        state = torch.load(weights_path, map_location=torch.device(device))
        model.load_state_dict(state['state_dict'])

    # Prep the loss file, read in current contents if any and then overwrite
    loss_values = []

    # training loop
    best_epoch_buffer, best_loss_value_buffer = 0, 0
    for epoch in range(start_epoch, nb_train_epoch):
        epoch_loss = 0
        for batch in tqdm(dl):

            # Constitute input and target with volumes extrated from split1_volume and split2_volume
            input = torch.concat([batch["split1_volume"][tio.DATA][0:dl.batch_size//2], batch["split2_volume"][tio.DATA][dl.batch_size//2:]], 0)
            target = torch.concat([batch["split2_volume"][tio.DATA][0:dl.batch_size//2], batch["split1_volume"][tio.DATA][dl.batch_size//2:]], 0)
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
        print(epoch_loss)

        # if the lowest train loss value so far is reached, save a checkpoint
        if best_loss_value_buffer == 0 or best_loss_value_buffer>epoch_loss:
            print("new best checkpoint obtained, saving checkpoint... ")
            Path(output_dir / f"minimum_train_loss_weights_epoch_{best_epoch_buffer}.torch").unlink(missing_ok=True)
            best_epoch_buffer, best_loss_value_buffer = int(epoch), float(epoch_loss)
            save_model(model, optimizer, epoch, output_dir / f"minimum_train_loss_weights_epoch_{best_epoch_buffer}.torch")
           

        # Save network if the interval has been reached
        if epoch % save_interval == 0 or epoch==nb_train_epoch-1 or epoch < 10:
            print("saving interval reached (epoch n°{}), saving checkpoint...".format(epoch))
            save_model(model, optimizer, epoch, output_dir / f"weights_epoch_{epoch}.torch")
        # Recording epoch loss
        loss_values.append(epoch_loss.detach().cpu())

    return loss_values




def save_train_loss(loss_values, loss_fn, start_epoch):

    map_epoch_loss = dict(zip(list(range(len(loss_values))), loss_values))

    best_epoch = list(dict(sorted(map_epoch_loss.items(), key=lambda item: float(item[1]), reverse=True)))[-1]

    # Save train loss after epoch 10 (reduce effect of squashing values)
    plt.plot(loss_values[10:])
    plt.xticks(list(range(0,len(loss_values[10:]),5))) 

    plt.ylabel("Train_Loss")
    plt.xlabel("Epoch, (minimum reached at epoch = {}, corresponding_value : {})".format(best_epoch,map_epoch_loss[best_epoch]))
    plt.title(loss_fn.name) 
    plt.savefig(loss_fn / f"from_epoch{start_epoch+10}")

    # Save train loss for the first 10 epochs
    plt.clf()
    plt.plot(loss_values[0:10])
    plt.xticks(list(range(len(loss_values[0:10])))) 

    plt.ylabel("Train_Loss")
    plt.xlabel("Epoch, (minimum reached at epoch = {}, corresponding_value : {})".format(best_epoch,map_epoch_loss[best_epoch]))
    plt.title(loss_fn.name) 
    plt.savefig(loss_fn / f"from_epoch{start_epoch}")


params = parse_arguments()
output_dir = Path(params.output_dir)
loss_fn = Path(params.loss_fn)

output_dir.mkdir(exist_ok=True)
loss_fn.parent.mkdir(exist_ok=True)
loss_fn.mkdir(exist_ok=True)

#Device configuration parameters
cuda_devices = [i for i in range(torch.cuda.device_count())]

# Save the training parameters
par_file = open(output_dir / "params.json", 'w')
json.dump(dict(vars(params)), par_file)
par_file.close()
# Initialize and train the model
model = make_model(cuda_devices)

assert params.train_patch_size[-3]%model.module.n_blocks == 0 and params.train_patch_size[-1]%model.module.n_blocks == 0 and params.train_patch_size[-2]%model.module.n_blocks == 0

# Get the data loading pipeline
train_loader = datasets.get_train_patch_loader(params.dataset_name, params.train_patch_size, params.batch_size, params.nb_patch_per_epoch)


if params.loss_func == "MSE":
    loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=params.weight_decay, lr=params.lr)
loss_values = train_model(train_loader, model, loss_func, optimizer, params.save_interval, output_dir, params.weights_path, params.start_epoch, params.nb_train_epoch, torch.device("cuda:0"))

save_train_loss(loss_values, loss_fn, params.start_epoch)
