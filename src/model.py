# author: Peter Yang
# date: 2021-05-17

'''This script trains and saves the specified transfer learning model.

Usage: model.py --train_dir=<train_dir> --valid_dir=<valid_dir> [--model="densenet"] [--image_size=300] [--batch_size=8] [--epoch=50] [--model_save_dir=models/cnn_model.pt] [--seed=2021]

Options:
--train_dir=<train_dir>            Relative path to the training folder (path)
--valid_dir=<valid_dir>            Relative path to the validation folder (path)
--model=<model>                    Transfer learning model(string) [default: 'densenet']
--image_size=<image_size>          Image size after processing(integer) [default: 300]
--batch_size=<batch_size>          Batch size(integer) [default: 8]
--epoch=<epoch>                    Number of epochs(integer) [default: 50]
--model_save_dir=<model_save_dir>  Relative path to where the model will be saved (path) [default: models/cnn_model.pt]
--seed=<seed>                      random seed(integer) [default: 2021]
'''

from docopt import docopt
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets, utils
from cnn_utils import *
import os

opt = docopt(__doc__)

def main(train_dir, valid_dir, model, image_size, batch_size, epoch, model_save_dir, seed):

    IMAGE_SIZE = int(image_size)
    BATCH_SIZE = int(batch_size)

    TRAIN_DIR = str(train_dir)
    VALID_DIR = str(valid_dir)

    MODEL = str(model)
    EPOCH = int(epoch)
    MODEL_PATH = str(model_save_dir)
    SEED = int(seed)
    
    train_dataset, train_loader, valid_dataset, valid_loader = image_transformation_train(
        TRAIN_DIR = TRAIN_DIR, VALID_DIR = VALID_DIR, IMAGE_SIZE = IMAGE_SIZE, BATCH_SIZE = BATCH_SIZE
    )
    
    cnn_model = models.inception_v3(pretrained=True)
    cnn_model.fc = nn.Linear(2048, 1)
    cnn_model.aux_logits = False
    
    print(f"The model selected is {MODEL}.")
    print()

    # model building
    
    if MODEL == "inception":
        pass
    elif MODEL == "densenet":
        cnn_model = models.densenet121(pretrained=True)
        new_layers = nn.Sequential(OrderedDict([
            ('new1', nn.Linear(1024, 500)),
            ('relu', nn.ReLU()),
            ('new2', nn.Linear(500, 1))
        ]))
        cnn_model.classifier = new_layers
    elif MODEL == "resnet":
        cnn_model = models.resnet50(pretrained=True)
        cnn_model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))
    elif MODEL == "vgg":
        cnn_model = models.vgg16_bn(pretrained=True)
        num_features = cnn_model.classifier[6].in_features
        features = list(cnn_model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, 1)]) # Add our layer with 4 outputs
        cnn_model.classifier = nn.Sequential(*features) # Replace the model classifier
    
    # initialize weight
    torch.manual_seed(SEED)
    cnn_model.apply(weights_init);

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model.to(device);

    # best hyperparameters for densenet tuned by the team
    best_hyperparameters = {'lr': 5.474673900063529e-05, 'beta1': 0.7190195309642723}

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(),                            
                                    lr=best_hyperparameters.get("lr", 0.001),
                                        betas=(best_hyperparameters.get("beta1", 0.9), 0.999))
    if MODEL != 'densenet': # use default hyperparameters if not densenet
        optimizer = torch.optim.Adam(cnn_model.parameters())
    results = trainer(cnn_model, criterion, optimizer, train_loader, valid_loader, device, epochs=EPOCH)
    
    try:
        os.makedirs('models')
    except:
        pass
    torch.save(cnn_model.state_dict(), MODEL_PATH)     # save model at PATH

if __name__ == "__main__":
    main(opt["--train_dir"], opt["--valid_dir"], opt["--model"], opt["--image_size"], opt["--batch_size"], opt["--epoch"], opt["--model_save_dir"], opt["--seed"])
