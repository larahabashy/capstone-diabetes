import torch
import torchvision
from torch import nn, optim
import numpy as np
import pandas as pd
from torchvision import transforms, models, datasets, utils
from captum.attr import GradientShap
from captum.attr import visualization as viz

def weights_init(model_layer, verbose = False):
    """Initialize weights of each layer to make the results more reproducible"""
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
        
def trainer(model, criterion, optimizer, train_loader, valid_loader, device, epochs=5, patience=5, verbose=True):
    """Simple training wrapper for PyTorch network."""
    
    data_type = torch.float32
    valid_loss = []
    for epoch in range(epochs):  # for each epoch
        train_batch_loss = 0
        train_batch_acc = 0
        valid_batch_loss = 0
        valid_batch_acc = 0
        
        # Training
        for X, y in train_loader:
            if device.type == 'cuda':
                X, y = X.to(device,data_type), y.to(device,data_type)
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
                if device.type == 'cuda':
                    X, y = X.to(device,data_type), y.to(device,data_type)
                y_hat = model(X).flatten()
                y_hat_labels = torch.sigmoid(y_hat) > 0.5
                loss = criterion(y_hat, y.type(data_type))
                valid_batch_loss += loss.item()
                valid_batch_acc += (y_hat_labels == y).type(data_type).sum().item()
        valid_accuracy = valid_batch_acc / len(valid_loader.dataset)
        valid_loss.append(valid_batch_loss / len(valid_loader))        
        
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(valid_loader):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model(inputs).flatten()
                preds = torch.sigmoid(outputs) > 0.5
                for t, p in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
        recall = confusion_matrix.numpy()[1,1]/(confusion_matrix.numpy()[1,1]+confusion_matrix.numpy()[1,0])
        model.train()
        
        # Early stopping
        if epoch > 0 and valid_loss[-1] > valid_loss[-2]:
            consec_increases += 1
        else:
            consec_increases = 0
        if consec_increases == patience:
            print(f"Stopped early at epoch {epoch + 1} - val loss increased for {consec_increases} consecutive epochs!")
            break
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch + 1}:",
                  f"Train Accuracy: {train_accuracy:.2f}.",
                  f"Valid Accuracy: {valid_accuracy:.2f}.",
                  f"Valid Recall: {recall:.2f}")
    
    return {"train_accuracy": train_accuracy, "valid_accuracy": valid_accuracy, "Valid_recall":recall}

def cnn_feature_importance(cnn, image_tensor):
    """visualize feature importance of CNN"""
    torch.manual_seed(2020); np.random.seed(2020)
    cnn = cnn.cpu()
    gradient_shap = GradientShap(cnn)
    rand_img_dist = torch.cat([image_tensor * 0, image_tensor * 1])
    attributions_gs = gradient_shap.attribute(image_tensor,
                                              n_samples=20,
                                              stdevs=0.15,
                                              baselines=rand_img_dist,
                                              target=0)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().detach().numpy(), (1,2,0)),
                                        np.transpose(image_tensor.squeeze().detach().numpy(), (1,2,0)),
                                        ["original_image", "blended_heat_map"],
                                        ["all", "absolute_value"],
                                        titles = ["Original Image", "Gradient SHAP"],
                                        cmap="plasma",
                                        show_colorbar=True,
                                        fig_size=(6, 6),
                                        alpha_overlay=0.7)

def get_test_accuracy(cnn, data_loader):
    """return accuracy on a holdout sample for a pytorch cnn model"""
    test_batch_acc = 0
    cnn.eval()
    with torch.no_grad():
        for X, y in data_loader:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                X, y = X.to(device), y.to(device)
            y_hat = cnn(X).flatten()
            y_hat_labels = torch.sigmoid(y_hat) > 0.5
            test_batch_acc += (y_hat_labels == y).type(torch.float32).sum().item()
    test_accuracy = test_batch_acc / len(data_loader.dataset)
    print(f"Test accuracy is {test_accuracy*100:.2f}%.")
    return test_accuracy

def pytorch_confusion_matrix(cnn, data_loader):
    """return confusion matrix on a holdout sample for a pytorch cnn model(binary classification)"""
    confusion_matrix = torch.zeros(2, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn.eval()
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = cnn(inputs).flatten()
            preds = torch.sigmoid(outputs) > 0.5
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    cm = pd.DataFrame(confusion_matrix.numpy(), columns = ['predict negative','predict positive'], index = ['actual negative', 'actual positive'])
    return cm

def recall_calculation(confusion_matrix):
    """return recall of holdout sample from a confusion matrix"""
    test_recall = confusion_matrix.iloc[1,1]/(confusion_matrix.iloc[1,1]+confusion_matrix.iloc[1,0])
    print(f"Test recall is {test_recall*100:.2f}%.")
    return test_recall


def image_transformation_train(TRAIN_DIR, VALID_DIR, IMAGE_SIZE, BATCH_SIZE):
    """image transformation on training, validation set"""
    train_transforms = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataset = torchvision.datasets.ImageFolder(root=VALID_DIR, transform=valid_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_dataset, train_loader, valid_dataset, valid_loader

def image_transformation_test(TEST_DIR, IMAGE_SIZE, BATCH_SIZE):
    """image transformation on test set"""

    test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    test_dataset = torchvision.datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_dataset, test_loader

def confusion_matrix_eval(cnn, data_loader):
    """retrieves false positives and false negatives for further investigation"""
    fp = []
    fn = []
    cnn.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = cnn(inputs).flatten()
            preds = torch.sigmoid(outputs) > 0.5
            j = 0 
            for t, p in zip(classes.view(-1), preds.view(-1)):
                if [float(t.cpu().numpy()), float(p.long().cpu().numpy())] == [0.0, 1.0]:
                    fp.append(data_loader.dataset.samples[j][0])
                elif [float(t.cpu().numpy()), float(p.long().cpu().numpy())] == [1.0, 0.0]:
                    fn.append(data_loader.dataset.samples[j][0])
                j += 1
        return fp, fn
    