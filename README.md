# TinyVGG from Scratch in PyTorch

This repository contains a Jupyter Notebook implementation of the TinyVGG (also known as VGG-tiny) image classification model, built from scratch using PyTorch.

The model is trained on a custom food image dataset to classify images into four categories:
* Chocolate Cake
* Cupcakes
* Macarons
* Sushi

## Project Overview

The notebook guides you through the entire process of building a deep learning model:

1.  **Data Preparation:**
    * Downloads and unzips the food image dataset from an S3 bucket.
    * Applies data augmentation (`TrivialAugmentWide`) and transformations (`Resize`, `ToTensor`) using `torchvision.transforms`.
    * Loads the data into training and testing sets using `datasets.ImageFolder`.
    * Creates `DataLoader` instances for batch processing (`BATCH_SIZE=32`).

2.  **Model Architecture (TinyVGG):**
    * The `TinyVGG` model is defined as a custom `nn.Module`.
    * It takes `(3, 64, 64)` images as input.
    * The architecture consists of two convolutional blocks followed by a classifier:
        * **Conv Block 1:** Conv2d (3->10) -> ReLU -> Conv2d (10->10) -> ReLU -> MaxPool2d
        * **Conv Block 2:** Conv2d (10->10) -> ReLU -> Conv2d (10->10) -> ReLU -> MaxPool2d
        * **Classifier:** Flatten -> Linear (1690 -> 4)
    * The model utilizes GPU (CUDA) if available.

3.  **Training & Evaluation:**
    * The model is trained for **50 epochs**.
    * **Loss Function:** `nn.CrossEntropyLoss()`
    * **Optimizer:** `torch.optim.Adam(lr=0.001)`
    * Includes helper functions for `train_step`, `test_step`, and a main `train()` function to handle the training and evaluation loop.
    * Results (accuracy and loss) for both training and testing are printed after each epoch.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Dependencies:**
    Make sure you have PyTorch and other required libraries installed.
    ```bash
    pip install torch torchvision torchinfo matplotlib numpy requests
    ```

3.  **Run the Notebook:**
    Open and run the `https://colab.research.google.com/drive/10YlhUO5ZrVoHNL5z9jNgZbWv03QCAmBQ` notebook using Jupyter or Google Colab. The dataset will be downloaded automatically when you run the cells.

## Dataset

* **Source:** The notebook downloads the dataset from:
    `https://programmingoceanacademy.s3.ap-southeast-1.amazonaws.com/image_classification_dataset.zip`
* **Structure:**
    * **Train (1000 images):** 250 images per class
    * **Test (300 images):** 75 images per class
* **Classes:** `chocolate_cake`, `cup_cakes`, `macarons`, `sushi`
