import torch
import torchvision
from torch import nn, optim
import numpy as np
import pandas as pd
from torchvision import transforms, models, datasets, utils
from captum.attr import GradientShap
from captum.attr import visualization as viz
from collections import OrderedDict
from collections.abc import Iterable

torch.manual_seed(2020)
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os


def weights_init(model_layer, verbose=False):
    """Initialize weights of each layer to make results more reproducible.

    Parameters
    ----------
    model_layer : torch.nn
        A layer of a pytorch model.
    verbose : bool, optional
        Prints the layer being initialized, by default False.
    """
    if isinstance(model_layer, nn.Conv2d):
        if verbose:
            print("Initializing weights of a Conv2d layer!")
        nn.init.normal_(model_layer.weight, mean=0, std=0.1)
        if model_layer.bias is not None:
            nn.init.zeros_(model_layer.bias)
    elif isinstance(model_layer, nn.BatchNorm2d):
        if verbose:
            print("Initializing weights of a batchnorm layer!")
        nn.init.ones_(model_layer.weight)
        nn.init.zeros_(model_layer.bias)
    elif isinstance(model_layer, nn.Linear):
        if verbose:
            print("Initializing weights of a Linear layer!")
        nn.init.xavier_uniform_(model_layer.weight)
        nn.init.zeros_(model_layer.bias)


def trainer(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    device,
    epochs=5,
    patience=5,
    verbose=True,
):
    """Simple training wrapper for PyTorch network.

    Parameters
    ----------
    model : torchvision.models
        A pytorch model.
    criterion : torch.nn
        A criterion for model training.
    optimizer : torch.optim
        An optimizer for model training.
    train_loader : torch.utils.data.DataLoader
        DataLoader object with data from the training set.
    valid_loader : torch.utils.data.DataLoader
        DataLoader object with data from the validation set.
    device : torch.device
        Device type CUDA or CPU.
    epochs : int, optional
        Number of epochs to train for, by default 5
    patience : int, optional
        Threshold for early stopping, by default 5
    verbose : bool, optional
        Print accuracy and recall scores for each epoch, by default True

    Returns
    -------
    dict
        A dictionary containing the training and validation set accuracies and validation set recall scores.

    Raises
    ------
    TypeError
        Raised if train_loader is not an iterable object.
    TypeError
        Raised if valid_loader is not an iterable object.
    TypeError
        Raised if epochs is not an integer.
    TypeError
        Raised if patience is not an integer.
    TypeError
        Raised if verbose is not a boolean.

    """
    if not isinstance(train_loader, Iterable):
        raise ValueError("train_loader is not iterable")

    if not isinstance(valid_loader, Iterable):
        raise ValueError("valid_loader is not iterable")

    if not isinstance(epochs, int):
        raise ValueError("Epochs is not an integer")

    if not isinstance(patience, int):
        raise ValueError("Patience is not an integer")

    if not isinstance(verbose, bool):
        raise ValueError("Vebose is not a boolean")

    data_type = torch.float32
    valid_loss = []
    for epoch in range(epochs):  # for each epoch
        train_batch_loss = 0
        train_batch_acc = 0
        valid_batch_loss = 0
        valid_batch_acc = 0

        # Training
        for X, y in train_loader:
            if device.type == "cuda":
                X, y = X.to(device, data_type), y.to(device, data_type)
            optimizer.zero_grad()
            y_hat = model(X).flatten()
            y_hat_labels = torch.sigmoid(y_hat) > 0.5
            loss = criterion(y_hat, y.type(data_type))
            loss.backward()
            optimizer.step()
            train_batch_loss += loss.item()
            train_batch_acc += (y_hat_labels == y).type(data_type).sum().item()
        train_accuracy = train_batch_acc / len(train_loader.dataset)

        # Validation
        confusion_matrix = torch.zeros(2, 2)
        model.eval()
        with torch.no_grad():
            for X, y in valid_loader:
                if device.type == "cuda":
                    X, y = X.to(device, data_type), y.to(device, data_type)
                y_hat = model(X).flatten()
                y_hat_labels = torch.sigmoid(y_hat) > 0.5
                loss = criterion(y_hat, y.type(data_type))
                valid_batch_loss += loss.item()
                valid_batch_acc += (y_hat_labels == y).type(data_type).sum().item()
        valid_accuracy = valid_batch_acc / len(valid_loader.dataset)
        valid_loss.append(valid_batch_loss / len(valid_loader))

        with torch.no_grad():
            for i, (inputs, classes) in enumerate(valid_loader):
                inputs = inputs.to(device, data_type)
                classes = classes.to(device, data_type)
                outputs = model(inputs).flatten()
                preds = torch.sigmoid(outputs) > 0.5
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        if (confusion_matrix.numpy()[1, 1] + confusion_matrix.numpy()[1, 0]) == 0:
            recall = 0
        else:
            recall = confusion_matrix.numpy()[1, 1] / (
                confusion_matrix.numpy()[1, 1] + confusion_matrix.numpy()[1, 0]
            )

        model.train()

        # Early stopping
        if epoch > 0 and valid_loss[-1] > valid_loss[-2]:
            consec_increases += 1
        else:
            consec_increases = 0
        if consec_increases == patience:
            print(
                f"Stopped early at epoch {epoch + 1} - val loss increased for {consec_increases} consecutive epochs!"
            )
            break

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch + 1}:",
                f"Train Accuracy: {train_accuracy:.2f}.",
                f"Valid Accuracy: {valid_accuracy:.2f}.",
                f"Valid Recall: {recall:.2f}",
            )

    return {
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "Valid_recall": recall,
    }


def get_test_accuracy(cnn, data_loader):
    """Returns the accuracy scores on a holdout sample for a pytorch cnn model.

    Parameters
    ----------
    cnn : torchvision.models
        A pytorch model.
    data_loader : torch.utils.data.DataLoader
        A dataloader iterating through the holdout test sample.

    Returns
    -------
    int
        Accuracy score on the holdout test sample.

    Raises
    ------
    ValueError
        Raised if data_loader is not iterable.
    """

    if not isinstance(data_loader, Iterable):
        raise ValueError("data_loader is not iterable")

    test_batch_acc = 0
    cnn.eval()
    with torch.no_grad():
        for X, y in data_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                X, y = X.to(device), y.to(device)
            y_hat = cnn(X).flatten()
            y_hat_labels = torch.sigmoid(y_hat) > 0.5
            test_batch_acc += (y_hat_labels == y).type(torch.float32).sum().item()
    test_accuracy = test_batch_acc / len(data_loader.dataset)
    print(f"Test accuracy is {test_accuracy*100:.2f}%.")
    return test_accuracy


def pytorch_confusion_matrix(cnn, data_loader):
    """Returns the confusion matrix on a holdout sample for a pytorch cnn model(binary classification).

    Parameters
    ----------
    cnn : torchvision.models
        A pytorch model.
    data_loader : torch.utils.data.DataLoader
        A dataloader iterating through the holdout test sample.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe containing the confusion matrix.

    Raises
    ------
    ValueError
        Raised when dataloader is not iterable.
    """

    if not isinstance(data_loader, Iterable):
        raise ValueError("data_loader is not iterable")

    confusion_matrix = torch.zeros(2, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.eval()
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = cnn(inputs).flatten()
            preds = torch.sigmoid(outputs) > 0.5
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    cm = pd.DataFrame(
        confusion_matrix.numpy(),
        columns=["predict negative", "predict positive"],
        index=["actual negative", "actual positive"],
    )
    return cm


def recall_calculation(confusion_matrix):
    """Returns the recall score for the holdout test sample from a confusion matrix.

    Parameters
    ----------
    confusion_matrix : pd.DataFrame
        A pandas dataframe with the confusion matrix for the holdout test sample.

    Returns
    -------
    int
        The recall score for the holdout test sample.

    Raises
    ------
    TypeError
        Error raised when input is not a pd.DataFrame.
    """

    if not isinstance(confusion_matrix, pd.DataFrame):
        raise TypeError("confusion_matrix needs to be a dataframe")

    if (confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 0]) == 0:
        test_recall = 0
    else:
        test_recall = confusion_matrix.iloc[1, 1] / (
            confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 0]
        )

    print(f"Test recall is {test_recall*100:.2f}%.")
    return test_recall


class LipoDataset(torch.utils.data.Dataset):
    """Define a custom dataset to be able to do albument transformations

    Parameters
    ----------
    torch.utils.data.Dataset : torch.utils.data.Dataset
        A torch dataset object.
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir

        # Create a list of filepaths of images and the respective label
        self.samples = []

        for i in os.listdir(root_dir):
            if i in ["positive", "negative"]:
                folder = os.path.join(root_dir, i)
                target = folder.split("/")[-1]
                for label in os.listdir(folder):
                    if str(label) not in [".DS_Store"]:
                        filepath = os.path.join(folder, label)
                        self.samples.append((target, filepath))

    def __len__(self):
        # Get the length of the samples
        return len(self.samples)

    def __getitem__(self, index):
        # Implement logic to get an image and its label using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `label` should be an integer in the range [0, model.num_classes - 1] where `model.num_classes`
        # is a value set in the `search.yaml` file.

        # get the filepath of the image based on the index and convert it to
        # color scale and then into a numpy array
        image = np.array(Image.open(self.samples[index][1]).convert("RGB"))

        # maps a label to an integer value
        label_to_int = {"positive": 1, "negative": 0}
        label = label_to_int[self.samples[index][0]]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            image = image / 255

        return image, label


def image_transformation_train(TRAIN_DIR, VALID_DIR, IMAGE_SIZE, BATCH_SIZE):
    """Apply transformations to images in training and validation directories.

    Parameters
    ----------
    TRAIN_DIR : str
        Path to directory containing train images.
    VALID_DIR : str
        Path to directory containing validation images.
    IMAGE_SIZE : int
        Size for images to be re-scaled to.
    BATCH_SIZE : int
        Batch size for train and validation loaders.

    Returns
    -------
    LipoDataset, DataLoader
        Return train_dataset and valid_dataset (LipoDataset objects) and train_loader and valid_loder (DataLoader objects).

    Raises
    ------
    TypeError
        Raised if IMAGE_SIZE is not an integer.
    TypeError
        Raised if BATCH_SIZE is not an integer.
    """

    if not isinstance(IMAGE_SIZE, int):
        raise TypeError("image size should be an integer")

    if not isinstance(BATCH_SIZE, int):
        raise TypeError("batch size should be an integer")

    train_transforms = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(0, 1.0)
            ),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            ToTensorV2(),
        ]
    )

    valid_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            ToTensorV2(),
        ]
    )

    train_dataset = LipoDataset(root_dir=TRAIN_DIR, transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_dataset = LipoDataset(root_dir=VALID_DIR, transform=valid_transforms)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    return train_dataset, train_loader, valid_dataset, valid_loader


def image_transformation_test(TEST_DIR, IMAGE_SIZE, BATCH_SIZE):
    """Applies Image transformation on test set.

    Parameters
    ----------
    TEST_DIR : str
        Path to directory containing test images.
    IMAGE_SIZE : int
        Size for images to be re-scaled to.
    BATCH_SIZE : int
        Batch size for train and validation loaders.

    Returns
    -------
    LipoDataset, DataLoader
        Returns the test_dataset (LipoDataset object) and the test_loader (DataLoader Object).

    Raises
    ------
    TypeError
        Raised if IMAGE_SIZE is not an integer.
    TypeError
        Raised if BATCH_SIZE is not an integer.
    """

    if not isinstance(IMAGE_SIZE, int):
        raise TypeError("image size should be an integer")

    if not isinstance(BATCH_SIZE, int):
        raise TypeError("batch size should be an integer")

    test_transforms = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=TEST_DIR, transform=test_transforms
    )
    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            ToTensorV2(),
        ]
    )

    test_dataset = LipoDataset(root_dir=TEST_DIR, transform=test_transforms)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return test_dataset, test_loader


def confusion_matrix_eval(cnn, data_loader):
    """Retrieves false positives and false negatives for further investigation

    Parameters
    ----------
    cnn : torchvision.models
        A trained pytorch model.
    data_loader : torch.utils.data.DataLoader
        A dataloader iterating through the holdout test sample.

    Returns
    -------
    dict
        Dictionary containing cases model classified as false positives and false negatives.

    Raises
    ------
    ValueError
        Raised if data_loader is not iterable.
    """

    if not isinstance(data_loader, Iterable):
        raise ValueError("data_loader is not iterable")

    fp = []
    fn = []
    cnn.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = cnn(inputs).flatten()
            preds = torch.sigmoid(outputs) > 0.5
            j = 0
            for t, p in zip(classes.view(-1), preds.view(-1)):
                if [float(t.cpu().numpy()), float(p.long().cpu().numpy())] == [
                    0.0,
                    1.0,
                ]:
                    fp.append(
                        data_loader.dataset.samples[(i * data_loader.batch_size + j)][1]
                    )
                elif [float(t.cpu().numpy()), float(p.long().cpu().numpy())] == [
                    1.0,
                    0.0,
                ]:
                    fn.append(
                        data_loader.dataset.samples[(i * data_loader.batch_size + j)][1]
                    )
                j += 1
        return {"false_positives": fp, "false_negatives": fn}


def make_model():
    """Make densenet model for transfer learning.

    Returns
    -------
    torchvision.models
        A densenet model ready to be used for transfer learning.
    """
    densenet = models.densenet121(pretrained=True)
    new_layers = torch.nn.Sequential(
        OrderedDict(
            [
                ("new1", torch.nn.Linear(1024, 500)),
                ("relu", torch.nn.ReLU()),
                ("new2", torch.nn.Linear(500, 1)),
            ]
        )
    )

    densenet.classifier = new_layers
    torch.manual_seed(2020)
    densenet.apply(weights_init)

    return densenet


def train(model, train_loader, hyperparameters, epochs=20):
    """Training wrapper for PyTorch network.

    Parameters
    ----------
    model : torchvision.models
        A pytorch model.
    train_loader : torch.utils.data.DataLoader
        DataLoader object with data from the training set.
    hyperparameters : dict
        A dictionary containing hyperparameter values for learning rate(lr) and beta1.
    epochs : int, optional
        The number of epochs to train model for, by default 20.

    Returns
    -------
    torchvision.models
        Returns trained model.

    Raises
    ------
    TypeError
        Raised if train_loader is not an iterable object.
    TypeError
        Raised if hyperparameters is not a dictionary.
    ValueError
        Raised if there are more than two keys in hyperparameters.
    """

    if not isinstance(train_loader, Iterable):
        raise TypeError("train_loader is not iterable")

    if not isinstance(hyperparameters, dict):
        raise TypeError("Hyparameters needs to be in a dictionary")

    if len(hyperparameters) != 2:
        raise ValueError("There are only two parameters wanted, lr and beta1")

    criterion = nn.BCEWithLogitsLoss()
    print("crit")
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters.get("lr", 0.001),
        betas=(hyperparameters.get("beta1", 0.9), 0.999),
    )
    print("optimizer")
    for epoch in range(epochs):
        for X, y in train_loader:
            if device.type == "cuda":
                X, y = X.to(device, torch.float32), y.to(device, torch.float32)
            print(X.shape)
            print(y)
            optimizer.zero_grad()
            y_hat = model(X).flatten()
            print(y_hat)
            loss = criterion(y_hat, y.type(torch.float32))
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, valid_loader):
    """Validation wrapper for PyTorch network.

    Parameters
    ----------
    model : torchvision.models
        A trained pytorch model to be evaluated on a validation set.
    valid_loader : torch.utils.data.DataLoader
        DataLoader object with data from the validation set.

    Returns
    -------
    int
        The average accuracy score on validation set.
    Raises
    ------
    ValueError
        Raised if valid_loader is not an iterable object.
    """

    if not isinstance(valid_loader, Iterable):
        raise ValueError("valid_loader is not iterable")

    model.eval()
    accuracy = 0
    with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
        for X, y in valid_loader:
            if device.type == "cuda":
                X, y = X.to(device, torch.float32), y.to(device, torch.float32)
            y_hat = model(X).flatten()
            y_hat_labels = torch.sigmoid(y_hat) > 0.5
            accuracy += (y_hat_labels == y).type(torch.float32).sum().item()
    accuracy /= len(valid_loader.dataset)  # avg accuracy
    print(f"Validation accuracy: {accuracy:.4f}")

    return accuracy
