import torch.utils.data
import os
from PIL import Image
import numpy as np


class SearchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir=os.path.join(os.path.dirname(__file__), "data/train"),
        transform=None,
    ):
        self.transform = transform
        # Implement additional initialization logic if needed
        self.root_dir = root_dir
        self.samples = []

        for i in os.listdir(root_dir):
            if i in ["positive", "negative"]:
                folder = os.path.join(root_dir, i)
                target = folder.split("/")[-1]
                for label in os.listdir(folder):
                    filepath = os.path.join(folder, label)
                    self.samples.append((target, filepath))

    def __len__(self):
        # Replace `...` with the actual implementation
        return len(self.samples)

    def __getitem__(self, index):
        # Implement logic to get an image and its label using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `label` should be an integer in the range [0, model.num_classes - 1] where `model.num_classes`
        # is a value set in the `search.yaml` file.

        # get the filepath of the image based on the index and converts it to
        # only RGB channels and then into a numpy array
        image = np.array(Image.open(self.samples[index][1]).convert("RGB"))

        # maps a label to an integer value
        label_to_int = {"positive": 1, "negative": 0}
        label = label_to_int[self.samples[index][0]]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


# print(os.path.join(os.path.dirname(__file__), "data/train"))
