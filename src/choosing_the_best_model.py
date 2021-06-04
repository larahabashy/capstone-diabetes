# Javairia Raza
# 2021-06-02

# This simple script produce visual results for the models for final report

import pandas as pd
import numpy as np
from cnn_utils import *
import altair as alt
import os

# get the recall and accuracies from the respective csv files in results folder

models_list = ["densenet", "inception", "vgg", "resnet"]
acc_models_list = []
recall_models_list = []

for model in models_list:
    df = pd.read_csv("results/" + model + "_test_accuracy.csv")
    acc_models_list.append(df["0"][0])

    df2 = pd.read_csv("results/" + model + "_test_recall.csv")
    recall_models_list.append(df2["0"][0])

test_summary_df = pd.DataFrame(
    {
        "model": ["DenseNet", "Inception", "VGG16", "ResNet"],
        "test_accuracy": acc_models_list,
        "test_recall": recall_models_list,
    }
)

# # create bar chart for accuracy

model_acc_bar_chart = (
    alt.Chart(test_summary_df)
    .mark_bar()
    .encode(
        x=alt.X("test_accuracy", title="Test Accuracy"),
        y=alt.Y("model", title="Model", sort="x"),
    )
)


model_acc_bar_chart.save("image/model_acc_bar_chart.png")

# # create bar chart for recall

model_recall_bar_chart = (
    alt.Chart(test_summary_df)
    .mark_bar()
    .encode(
        x=alt.X("test_recall", title="Test Recall"),
        y=alt.Y("model", title="Model", sort="x"),
    )
)

model_recall_bar_chart.save("image/model_recall_bar_chart.png")

# get file sizes

file_paths = [
    "models/trained_models_May25/resnet.pt",
    "models/trained_models_May25/inception_bo_simple.pth",
    "models/trained_models_June2/densenet_final.pth",
    "models/trained_models_June2/vgg16-final.pth",
]

size_df = pd.DataFrame(
    {
        "Model": ["ResNet", "Inception", "DenseNet", "VGG16"],
        "Size (MB)": [
            np.round(os.path.getsize(file) / 1000000, 2) for file in file_paths
        ],  # gets size and converts bytes to MB
    }
)
size_df = size_df.sort_values("Size (MB)")
size_df.to_csv("results/models_size_comparison", index=False)  # saves df to results
