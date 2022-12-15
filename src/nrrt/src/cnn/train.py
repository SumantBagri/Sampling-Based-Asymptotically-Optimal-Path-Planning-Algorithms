#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
from torch import optim as optim
from map_dataset import MapDataset
from nrrt_cnn import NRRTCNN
from tqdm import tqdm

INPUT_DIR = "map_dataset"
LABEL_DIR = "map_labels"

IMG_SIZE = 256
BATCH_SIZE = 10


if __name__ == "__main__":
    map_types = ["alternating_gaps", "forest", "mazes", "shifting_gaps", "bugtrap_forest", 
                 "gaps_and_forest", "multiple_bugtraps", "single_bugtrap"]
    
    map_list = []
    label_list = []

    # model parameters
    clearance = 1
    stepsize = 1

    # model hyperparameters
    lr = 1e-4
    beta_1 = 0.9
    beta_2 = 0.999
    num_epochs = 50
    model_num = 7

    print('collating training data')

    # collate training data
    for map_type in map_types:
        # list of files for training data
        labels = os.listdir('{}/{}'.format(LABEL_DIR, map_type))

        # build full paths for maps/labels
        maps = ["{}/{}/train/{}.png".format(INPUT_DIR, map_type, lab.split('_')[0]) for lab in labels]
        labels = ["{}/{}/{}".format(LABEL_DIR, map_type, lab) for lab in labels]
        labels = ["{}/{}/{}".format(LABEL_DIR, map_type, lab) for lab in labels]

        map_list += maps
        label_list += labels
    
    print('loading training data')

    train_map_list = map_list[:int(0.8*len(map_list))]
    train_label_list = label_list[:int(0.8*len(label_list))]

    test_map_list = map_list[int(0.8*len(map_list)):]
    test_label_list = label_list[int(0.8*len(label_list)):]

    # load dataset
    train_dataset = MapDataset(train_map_list, train_label_list, map_types, IMG_SIZE)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = MapDataset(test_map_list, test_label_list, map_types, IMG_SIZE)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("size of training dataset: {}".format(len(train_dataset)))
    print("size of training dataset: {}".format(len(test_dataset)))
    print('creating model')

    # create model    
    dev = torch.device("cuda")
    model = NRRTCNN(clearance, stepsize).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    loss_fn = nn.BCELoss()
    tq_obj = tqdm(range(num_epochs), desc="Model Training")
    
    train_losses = []
    test_losses = []

    print('prep for training complete')

    for it in tq_obj:
        running_loss_train = 0
        running_loss_test = 0

        model.train()
        
        for img, lab in train_loader:
            img = img[:, None, :, :].to(dev)
            lab = lab[:, None, :, :].to(dev)
            opt.zero_grad()

            # generate output and back propagate
            output = model(img)
            loss = loss_fn(output, lab)
            loss.backward()
            opt.step()
            
            running_loss_train += loss.item()
        
        avg_loss_train = running_loss_train / len(train_loader)
        train_losses.append(avg_loss_train)
 
        model.eval()

        for img, lab in test_loader:
            img = img[:, None, :, :].to(dev)
            lab = lab[:, None, :, :].to(dev)

            output = model(img)
            loss = loss_fn(output, lab)
            running_loss_test += loss.item()

        avg_loss_test = running_loss_test / len(test_loader)
        test_losses.append(avg_loss_test)

        #if it % 5 == 0:
        tqdm.write(f"Epoch: {it}  Training Loss: {avg_loss_train} Testing Loss: {avg_loss_test}")

        # save intermediate models
        if (it+1) % 10 == 0:
            state = {
                'epoch': it,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'train_loss': train_losses,
                'test_loss': test_losses
            }
            torch.save(state, 'model/model_{:03d}_{:03d}.pth'.format(model_num, it+1))

