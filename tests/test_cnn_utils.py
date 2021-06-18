# gets the path for the src directory to access cnn_utils script
import sys
import os

from numpy.core.numeric import NaN

sys.path.append(os.getcwd() + "/src")

# other imports
from cnn_utils import *
import pytest
from pytest import raises
import pandas as pd
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets, utils
from collections.abc import Iterable
import random
from math import isclose


torch.manual_seed(2020)
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up the custom toy dataset for testing
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.samples = data

    def __getitem__(self, index):
        image = self.samples[index]
        label = 0
        return image, label  # data is all negative

    def __len__(self):
        return len(self.samples)


x = torch.rand(size=(25, 3, 300, 300), dtype=torch.float32)
train, valid, test = torch.utils.data.random_split(
    x, [17, 4, 4], torch.Generator().manual_seed(2020)
)

train_dataset = ToyDataset(train)
valid_dataset = ToyDataset(valid)
test_dataset = ToyDataset(test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)


# directories used for testing
# download trained_models and data_split_unmapped directories from OneDrive
MODEL_DIR = "trained_models/trained_models_June2/densenet_final.pth"
train_path = "data_split_unmapped/train/"
valid_path = "data_split_unmapped/train/"
test_path = "data_split_unmapped/test/"

model = make_model()
params_before = torch.cat([param.view(-1) for param in model.parameters()])
model.load_state_dict(torch.load(MODEL_DIR, map_location=torch.device("cpu")))
params_after = torch.cat([param.view(-1) for param in model.parameters()])

# tests
def test_make_model_has_correct_layers():
    assert model.state_dict()["classifier.new1.weight"].shape[1] == 1024
    assert model.state_dict()["classifier.new1.bias"].shape[0] == 500
    assert model.state_dict()["classifier.new2.weight"].shape[1] == 500
    assert model.state_dict()["classifier.new2.bias"].shape[0] == 1


def test_model_weights_are_loaded():
    assert torch.equal(params_before, params_after) == False


def test_model_is_predicting():
    model.eval()
    img, label = next(iter(test_loader))
    y_pred = torch.sigmoid(model(img).flatten())
    assert sum([0 < value < 1 for value in y_pred]) == len(y_pred)


def test_model_weights_are_intialized_the_same_everytime():
    model1 = make_model()
    params_1 = torch.cat([param.view(-1) for param in model1.parameters()])
    model2 = make_model()
    params_2 = torch.cat([param.view(-1) for param in model2.parameters()])
    assert torch.equal(params_1, params_2)


def test_weights_init_bias_layer_all_zeros():
    model = make_model()
    assert torch.equal(torch.zeros([64]), model.state_dict()["features.norm0.bias"])


def test_trainer_output_accuracies():
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    dict_of_acc = trainer(
        model,
        criterion,
        optimizer,
        train_loader,
        valid_loader,
        device,
        epochs=1,
        patience=5,
    )
    assert np.round(dict_of_acc["train_accuracy"]) == 1.0
    assert dict_of_acc["valid_accuracy"] == 1.0
    assert dict_of_acc["Valid_recall"] == 0


def test_image_transformation_train_input():
    not_int_image_size = 100.3
    with raises(TypeError):
        image_transformation_train(train_path, valid_path, not_int_image_size, 8)

    not_int_batch_size = 7.3
    with raises(TypeError):
        image_transformation_train(train_path, valid_path, 300, not_int_batch_size)


def test_image_transformation_test_input():
    not_int = 1.3
    with raises(TypeError):
        image_transformation_test(test_path, 300, not_int)


def test_image_transformation_train_resizing_correct():
    batch_size = 8
    image_size = 300
    results = image_transformation_train(train_path, valid_path, image_size, batch_size)
    images, labels = next(iter(results[1]))
    assert len(results[0]) == 246  # check total training images
    assert images.shape == torch.Size([8, 3, 300, 300])
    assert len(labels) == batch_size


def test_image_transformation_test_resizing_correct():
    batch_size = 8
    image_size = 300
    results = image_transformation_test(test_path, image_size, batch_size)
    images, labels = next(iter(results[1]))
    assert images.shape == torch.Size([8, 3, 300, 300])
    assert len(labels) == batch_size


def test_confusion_matrix_eval_input():
    not_valid_loader = torch.rand(size=(8, 3, 300, 300), dtype=torch.float32)
    with raises(ValueError):
        confusion_matrix_eval(model, not_valid_loader)


def test_confusion_matrix_eval_output():
    assert len(confusion_matrix_eval(model, test_loader)["false_positives"]) == 0
    assert len(confusion_matrix_eval(model, test_loader)["false_negatives"]) == 0


def test_get_test_accuracy_input():
    not_valid_loader = torch.rand(size=(8, 3, 300, 300), dtype=torch.float32)
    with raises(ValueError):
        get_test_accuracy(model, not_valid_loader)


def test_get_test_accuracy_output():
    assert np.round(get_test_accuracy(model, test_loader) * 100) == 100


def test_pytorch_confusion_martix_input():
    not_valid_loader = torch.rand(size=(8, 3, 300, 300), dtype=torch.float32)
    with raises(ValueError):
        pytorch_confusion_matrix(model, not_valid_loader)


def helper_confusion_matrix_df(arr):
    return pd.DataFrame(
        data=arr,
        columns=["predict negative", "predict positive"],
        index=["actual negative", "actual positive"],
        dtype=np.float32,
    )


def test_pytorch_confusion_martix_output():
    df = helper_confusion_matrix_df([[4.0, 0.0], [0.0, 0.0]])
    df_from_func = pytorch_confusion_matrix(model, test_loader)
    assert df.equals(df_from_func)
    assert df_from_func.to_numpy().sum() == len(test_dataset.samples)


def test_recall_calculation_input_should_be_dataframe():
    not_a_df = np.array([[4.0, 0.0], [0.0, 0.0]])
    with raises(TypeError):
        recall_calculation(not_a_df)


def test_recall_calculation_output():
    df = helper_confusion_matrix_df([[0.0, 0.0], [0.0, 4.0]])
    assert recall_calculation(df) == 1.0
    assert 0 <= recall_calculation(df) <= 1.0


def test_train_input():
    parameters = {"lr": 0.0003, "beta1": 0.83}
    not_train_loader = torch.rand(size=(8, 3, 300, 300), dtype=torch.float32)
    with raises(TypeError):
        train(model, not_train_loader, parameters, epochs=1)

    not_a_dict = [7e-5, 0.99]
    with raises(TypeError):
        train(model, train_loader, not_a_dict, epochs=1)

    assert isinstance(model, object)


def test_evaluate_output():
    assert evaluate(model, valid_loader) == 1.0


def test_evaluate_input():
    not_valid_loader = torch.rand(size=(8, 3, 300, 300), dtype=torch.float32)
    with raises(ValueError):
        evaluate(model, not_valid_loader)

    assert isinstance(model, object)
