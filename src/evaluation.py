# author: Peter Yang
# date: 2021-05-18

"""This script evaluates the performance of model on the test set.

Usage: evaluation.py --test_dir=<test_dir> [--model_dir=models/cnn_model.pt] [--model="densenet"] [--image_size=300] [--batch_size=8] [--seed=2021]

Options:
--test_dir=<test_dir>              Relative path to the test folder (path)
--model_dir=<model_dir>            Relative path to the previous saved model (path) [default: models/cnn_model.pt]
--model=<model>                    Transfer learning model(string) [default: 'densenet']
--image_size=<image_size>          Image size after processing(integer) [default: 300]
--batch_size=<batch_size>          Batch size(integer) [default: 8]
--seed=<seed>                      random seed(integer) [default: 2021]
"""

from docopt import docopt
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets, utils
import os
import shutil
from cnn_utils import *

opt = docopt(__doc__)


def main(test_dir, model_dir, model, image_size, batch_size, seed):
    IMAGE_SIZE = int(image_size)
    BATCH_SIZE = int(batch_size)

    TEST_DIR = str(test_dir)
    MODEL_DIR = str(model_dir)

    MODEL = str(model)
    SEED = int(seed)

    #load test data
    test_dataset, test_loader = image_transformation_test(
        TEST_DIR=TEST_DIR, IMAGE_SIZE=IMAGE_SIZE, BATCH_SIZE=BATCH_SIZE
    )

    #model building 

    cnn_model = models.inception_v3(pretrained=True)
    cnn_model.fc = nn.Linear(2048, 1)
    cnn_model.aux_logits = False

    if MODEL == "inception":
        pass
    elif MODEL == "densenet":
        cnn_model = models.densenet121(pretrained=True)
        new_layers = nn.Sequential(
            OrderedDict(
                [
                    ("new1", nn.Linear(1024, 500)),
                    ("relu", nn.ReLU()),
                    ("new2", nn.Linear(500, 1)),
                ]
            )
        )
        cnn_model.classifier = new_layers
    elif MODEL == "resnet":
        cnn_model = models.resnet50(pretrained=True)
        cnn_model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 1))
    elif MODEL == "vgg":
        cnn_model = models.vgg16_bn(pretrained=True)
        num_features = cnn_model.classifier[6].in_features
        features = list(cnn_model.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, 1)])  # Add our layer with 4 outputs
        cnn_model.classifier = nn.Sequential(*features)  # Replace the model classifier

    cnn_model.load_state_dict(
        torch.load(MODEL_DIR, map_location=torch.device("cpu"))
    )  # load model weights from PATH

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    try:
        os.mkdir("results")
    except:
        pass

    # get test accuracy
    test_accura = get_test_accuracy(cnn_model, test_loader)
    pd.DataFrame({test_accura}).to_csv(
        "results/test_accuracy.csv", index=False
    )
    print()

    # get test confusion matrix
    cmtx = pytorch_confusion_matrix(cnn_model, test_loader)
    print("Confusion matrix of the test set: ")
    print(cmtx)
    cmtx.to_csv("results/test_confusion_matrix.csv", index=False)
    print()

    # get test recall
    test_recall_num = recall_calculation(cmtx)
    pd.DataFrame({test_recall_num}).to_csv(
        "results/test_recall.csv", index=False
    )


if __name__ == "__main__":
    main(
        opt["--test_dir"],
        opt["--model_dir"],
        opt["--model"],
        opt["--image_size"],
        opt["--batch_size"],
        opt["--seed"],
    )
