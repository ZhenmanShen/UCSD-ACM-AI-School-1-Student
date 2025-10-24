# Code Walkthrough - CIFAR-10 Demo Notebook

This guide walks through the `cifar10_demo_student.ipynb` notebook **line by line**, explaining what each piece of code does. This is for students who don't understand Python code well.

---

## Table of Contents
1. [Cell 0: Introduction](#cell-0-introduction)
2. [Cell 1: Setup Markdown](#cell-1-setup-markdown)
3. [Cell 2: Import Libraries](#cell-2-import-libraries)
4. [Cell 3: Understanding the Data Markdown](#cell-3-understanding-the-data-markdown)
5. [Cell 4: Load CIFAR-10 Dataset](#cell-4-load-cifar-10-dataset)
6. [Cell 5: Visualize Images Markdown](#cell-5-visualize-images-markdown)
7. [Cell 6: Display Sample Images](#cell-6-display-sample-images)
8. [Cell 7: Feature Engineering Markdown](#cell-7-feature-engineering-markdown)
9. [Cell 8: Extract HOG Features](#cell-8-extract-hog-features)
10. [Cell 9: Train Classifier Markdown](#cell-9-train-classifier-markdown)
11. [Cell 10: Train Logistic Regression](#cell-10-train-logistic-regression)
12. [Cell 11: Data Augmentation Markdown](#cell-11-data-augmentation-markdown)
13. [Cell 12: Show Augmentation Examples](#cell-12-show-augmentation-examples)
14. [Cell 13: CNN Markdown](#cell-13-cnn-markdown)
15. [Cell 14: Define CNN Model](#cell-14-define-cnn-model)
16. [Cell 15: Prepare Data Markdown](#cell-15-prepare-data-markdown)
17. [Cell 16: Create Data Loaders](#cell-16-create-data-loaders)
18. [Cell 17: Training Markdown](#cell-17-training-markdown)
19. [Cell 18: Train the Model](#cell-18-train-the-model)
20. [Cell 19: Summary Markdown](#cell-19-summary-markdown)

---

## Cell 0: Introduction

This is just a markdown cell (text, not code). It tells you what the notebook is about.

```markdown
# CIFAR-10 Demo - Image Classification

Welcome to Computer Vision! In this demo, you'll learn how computers can recognize objects in images.

## What You'll Learn:
1. Image Classification Basics - Teaching computers to recognize objects
2. Feature Engineering - The old-school approach (manually designed features)
3. Data Augmentation - Making models work in different conditions
4. CNNs (Convolutional Neural Networks) - The modern deep learning approach
```

**What this does**: Just explains the notebook structure. No code to run.

---

## Cell 1: Setup Markdown

Another markdown cell explaining what libraries we're about to import.

```markdown
## Setup: Import Libraries

**What are these?**
- PyTorch: A deep learning framework (like TensorFlow)
- torchvision: Computer vision tools for PyTorch
- NumPy: For numerical operations
- Matplotlib: For visualizing images
- scikit-learn: Traditional machine learning tools
```

**What this does**: Describes what the libraries do. No code to run.

---

## Cell 2: Import Libraries

**This is our first code cell!** Let's break it down line by line.

```python
import torch
```
- **What it does**: Imports PyTorch, the deep learning library
- **Why we need it**: To build and train neural networks

```python
import torchvision
```
- **What it does**: Imports torchvision (computer vision tools for PyTorch)
- **Why we need it**: To load datasets and apply image transformations

```python
from torchvision import datasets, transforms
```
- **What it does**: Imports specific modules from torchvision
  - `datasets`: Contains pre-loaded datasets like CIFAR-10
  - `transforms`: Tools to modify images (resize, rotate, etc.)

```python
from torch.utils.data import DataLoader
```
- **What it does**: Imports DataLoader
- **Why we need it**: To load data in batches during training

```python
import torch.nn as nn
```
- **What it does**: Imports neural network modules
- **Why we need it**: To build neural network layers (Conv2d, Linear, etc.)

```python
import torch.optim as optim
```
- **What it does**: Imports optimization algorithms
- **Why we need it**: To update model weights during training (Adam, SGD, etc.)

```python
import matplotlib.pyplot as plt
```
- **What it does**: Imports matplotlib for plotting
- **Why we need it**: To visualize images and graphs

```python
import numpy as np
```
- **What it does**: Imports NumPy for numerical operations
- **Why we need it**: To work with arrays and do math operations

```python
from sklearn.linear_model import LogisticRegression
```
- **What it does**: Imports Logistic Regression from scikit-learn
- **Why we need it**: For the feature engineering example

```python
from skimage.feature import hog
```
- **What it does**: Imports HOG (Histogram of Oriented Gradients) feature extractor
- **Why we need it**: To extract edge features from images (old approach)

```python
from tqdm.notebook import tqdm
```
- **What it does**: Imports a progress bar for loops
- **Why we need it**: To show progress when processing images

```python
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
```
- **What it does**: Checks what hardware is available and selects it
  - If NVIDIA GPU (CUDA) is available → use GPU
  - Else if Apple Silicon GPU (MPS) is available → use MPS
  - Else use CPU
- **Why we need it**: GPUs make training much faster!

```python
print(f"Using device: {device}")
```
- **What it does**: Prints which device we're using
- **Example output**: `Using device: mps` or `Using device: cpu`

---

## Cell 3: Understanding the Data Markdown

Markdown cell explaining what CIFAR-10 is.

**What this does**: Just text explaining the dataset. No code.

---

## Cell 4: Load CIFAR-10 Dataset

```python
transform = transforms.Compose([transforms.ToTensor()])
```
- **What it does**: Creates a transformation pipeline
  - `transforms.Compose([...])`: Chains multiple transformations together
  - `transforms.ToTensor()`: Converts images (as PIL Images or NumPy arrays) to PyTorch tensors
- **Why we need it**: PyTorch models require input as tensors, not images

```python
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
```
- **What it does**: Loads the CIFAR-10 training dataset
  - `root='./data'`: Where to save the downloaded data
  - `train=True`: Load the training set (50,000 images)
  - `download=True`: Download if not already present
  - `transform=transform`: Apply the ToTensor transformation to each image

```python
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```
- **What it does**: Loads the CIFAR-10 test dataset
  - `train=False`: Load the test set (10,000 images)
  - Everything else is the same as above

```python
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
```
- **What it does**: Creates a list of class names
- **Why we need it**: To convert numeric labels (0-9) to human-readable names

```python
print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {classes}")
```
- **What it does**: Prints dataset information
  - `len(train_dataset)`: Number of training images
  - `len(test_dataset)`: Number of test images
- **Example output**:
  ```
  Training images: 50000
  Test images: 10000
  Classes: ['airplane', 'automobile', 'bird', ...]
  ```

---

## Cell 5: Visualize Images Markdown

Markdown cell explaining we're going to look at sample images.

**What this does**: Just text. No code.

---

## Cell 6: Display Sample Images

This cell is more complex! Let's break it down step by step.

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
```
- **What it does**: Creates a figure with a 2×5 grid of subplots
  - `2, 5`: 2 rows, 5 columns = 10 subplots total
  - `figsize=(12, 5)`: Figure size in inches (width=12, height=5)
  - `fig`: The whole figure
  - `axes`: A 2D array of subplot axes

```python
for idx, class_name in enumerate(classes):
```
- **What it does**: Loops through each class
  - `enumerate(classes)`: Gives us both the index (0, 1, 2, ...) and the class name
  - `idx`: 0 for 'airplane', 1 for 'automobile', etc.
  - `class_name`: 'airplane', 'automobile', etc.

```python
    for i in range(len(train_dataset)):
```
- **What it does**: Loops through all training images
- **Why**: We're searching for the first image of the current class

```python
        img, label = train_dataset[i]
```
- **What it does**: Gets the i-th image and its label
  - `img`: The image as a tensor (3×32×32)
  - `label`: The class index (0-9)

```python
        if label == idx:
```
- **What it does**: Checks if this image belongs to the current class we're looking for

```python
            ax = axes[idx // 5, idx % 5]
```
- **What it does**: Gets the subplot for this class
  - `idx // 5`: Integer division. Row number (0 or 1)
    - idx=0: 0//5 = 0 (row 0)
    - idx=5: 5//5 = 1 (row 1)
  - `idx % 5`: Remainder. Column number (0-4)
    - idx=0: 0%5 = 0 (col 0)
    - idx=7: 7%5 = 2 (col 2)

```python
            img_np = img.permute(1, 2, 0).numpy()
```
- **What it does**: Converts the image tensor for display
  - `img`: Shape is (3, 32, 32) - PyTorch format (Channels, Height, Width)
  - `permute(1, 2, 0)`: Rearranges to (32, 32, 3) - Matplotlib format (Height, Width, Channels)
  - `.numpy()`: Converts from PyTorch tensor to NumPy array

```python
            ax.imshow(img_np)
```
- **What it does**: Displays the image in this subplot

```python
            ax.set_title(class_name)
```
- **What it does**: Sets the title to the class name

```python
            ax.axis('off')
```
- **What it does**: Hides the x and y axes (makes it cleaner)

```python
            break
```
- **What it does**: Stops the inner loop (we found our image, no need to keep searching)

```python
plt.tight_layout()
```
- **What it does**: Adjusts spacing between subplots so they don't overlap

```python
plt.show()
```
- **What it does**: Displays the figure

```python
sample_img, _ = train_dataset[0]
```
- **What it does**: Gets the first image from the dataset
  - `sample_img`: The image
  - `_`: The label (we don't care about it, so we use `_`)

```python
print(f"\nImage shape: {sample_img.shape}")
```
- **What it does**: Prints the shape of the image
- **Example output**: `Image shape: torch.Size([3, 32, 32])`

```python
print("That's 3 color channels (RGB), 32 pixels tall, 32 pixels wide!")
```
- **What it does**: Explains what the shape means

---

## Cell 7: Feature Engineering Markdown

Markdown explaining the old approach to computer vision.

**What this does**: Just explanatory text. No code.

---

## Cell 8: Extract HOG Features

This cell extracts HOG (Histogram of Oriented Gradients) features from images.

```python
def extract_hog_features(dataset, num_samples=5000):
```
- **What it does**: Defines a function to extract HOG features
  - `dataset`: The dataset to process
  - `num_samples=5000`: How many images to process (default 5000)

```python
    """
    Extract HOG features from images

    HOG finds edges in images and describes them with numbers.
    Think of it like describing a person: "tall, brown hair, glasses"
    """
```
- **What it does**: Documentation string explaining what the function does
- **This is just a comment**: Doesn't execute as code

```python
    features = []
    labels = []
```
- **What it does**: Creates empty lists to store features and labels

```python
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Extracting features"):
```
- **What it does**: Loops through images with a progress bar
  - `min(num_samples, len(dataset))`: Process at most num_samples images
  - `tqdm(...)`: Shows a progress bar
  - `desc="Extracting features"`: Label for the progress bar

```python
        img, label = dataset[i]
```
- **What it does**: Gets the i-th image and its label

```python
        img_gray = img.mean(dim=0).numpy()
```
- **What it does**: Converts the color image to grayscale
  - `img.mean(dim=0)`: Averages across the color channels (dimension 0)
  - Original: (3, 32, 32) → After mean: (32, 32)
  - `.numpy()`: Converts to NumPy array

```python
        feat = hog(img_gray, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9)
```
- **What it does**: Extracts HOG features from the grayscale image
  - `pixels_per_cell=(4, 4)`: Divide image into 4×4 pixel cells
  - `cells_per_block=(2, 2)`: Group cells into 2×2 blocks for normalization
  - `orientations=9`: Use 9 orientation bins for gradients
  - **Output**: A 1D array of numbers describing edges

```python
        features.append(feat)
        labels.append(label)
```
- **What it does**: Adds the features and label to the lists

```python
    return np.array(features), np.array(labels)
```
- **What it does**: Converts lists to NumPy arrays and returns them
  - `features`: 2D array of shape (num_samples, 1764)
  - `labels`: 1D array of shape (num_samples,)

```python
print("Extracting HOG features from training data...")
X_train, y_train = extract_hog_features(train_dataset, num_samples=5000)
```
- **What it does**: Extracts HOG features from 5000 training images
  - `X_train`: Feature matrix (5000, 1764)
  - `y_train`: Labels array (5000,)

```python
print("Extracting HOG features from test data...")
X_test, y_test = extract_hog_features(test_dataset, num_samples=2000)
```
- **What it does**: Extracts HOG features from 2000 test images

```python
print(f"\nEach image is now represented by {X_train.shape[1]} numbers")
print("These numbers describe the edges and shapes in the image")
```
- **What it does**: Prints information about the features
- **Example output**: `Each image is now represented by 1764 numbers`

---

## Cell 9: Train Classifier Markdown

Markdown explaining we'll train a Logistic Regression classifier.

**What this does**: Just text. No code.

---

## Cell 10: Train Logistic Regression

```python
print("Training classifier on HOG features...")
clf = LogisticRegression(max_iter=100, verbose=1)
```
- **What it does**: Creates a Logistic Regression classifier
  - `max_iter=100`: Maximum 100 iterations for training
  - `verbose=1`: Print training progress

```python
clf.fit(X_train, y_train)
```
- **What it does**: Trains the classifier on the HOG features
  - `X_train`: Feature matrix (5000, 1764)
  - `y_train`: Labels (5000,)
  - This is where the learning happens!

```python
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
```
- **What it does**: Calculates accuracy on training and test sets
  - `.score()`: Returns the accuracy (percentage of correct predictions)

```python
print(f"\n{'='*50}")
print("FEATURE ENGINEERING RESULTS")
print(f"{'='*50}")
print(f"Train Accuracy: {train_acc:.1%}")
print(f"Test Accuracy:  {test_acc:.1%}")
print(f"{'='*50}")
```
- **What it does**: Prints the results in a nice format
  - `'='*50`: Prints 50 equal signs
  - `{train_acc:.1%}`: Formats as percentage with 1 decimal place
- **Example output**:
  ```
  ==================================================
  FEATURE ENGINEERING RESULTS
  ==================================================
  Train Accuracy: 81.4%
  Test Accuracy:  41.6%
  ==================================================
  ```

```python
print("\n💡 Not great! Manual features can only do so much.")
print("   This is why deep learning became popular!")
```
- **What it does**: Explains why this approach is limited

---

## Cell 11: Data Augmentation Markdown

Markdown explaining data augmentation.

**What this does**: Explanatory text about why augmentation matters. No code.

---

## Cell 12: Show Augmentation Examples

This cell shows what data augmentation looks like.

```python
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])
```
- **What it does**: Defines augmentation transformations
  - `RandomHorizontalFlip(p=1.0)`: Always flip (p=1.0 means 100% probability, just for demo)
  - `RandomRotation(15)`: Rotate by up to ±15 degrees
  - `ColorJitter(brightness=0.3, contrast=0.3)`: Randomly adjust brightness and contrast
  - `ToTensor()`: Convert to tensor

```python
original_img, label = train_dataset[100]
```
- **What it does**: Gets the 100th training image

```python
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
```
- **What it does**: Creates a 1×6 grid of subplots

```python
axes[0].imshow(original_img.permute(1, 2, 0))
axes[0].set_title("Original")
axes[0].axis('off')
```
- **What it does**: Shows the original image in the first subplot

```python
for i in range(1, 6):
```
- **What it does**: Loops 5 times to create 5 augmented versions

```python
    from PIL import Image
    pil_img = Image.fromarray((original_img.permute(1,2,0).numpy() * 255).astype(np.uint8))
```
- **What it does**: Converts tensor to PIL Image
  - `permute(1,2,0)`: (3,32,32) → (32,32,3)
  - `* 255`: Scale from [0,1] to [0,255]
  - `.astype(np.uint8)`: Convert to unsigned 8-bit integers
  - `Image.fromarray()`: Create PIL Image

```python
    aug_img = augment_transform(pil_img)
```
- **What it does**: Applies the augmentation transformations

```python
    axes[i].imshow(aug_img.permute(1, 2, 0))
    axes[i].set_title(f"Augmented {i}")
    axes[i].axis('off')
```
- **What it does**: Shows the augmented image

```python
plt.suptitle(f"Data Augmentation Example - {classes[label]}")
plt.tight_layout()
plt.show()
```
- **What it does**: Adds a super title and displays the figure

```python
print("\n💡 The model sees many variations of each image during training!")
print("   This helps it learn robust features that work in different conditions.")
```
- **What it does**: Explains the benefit

---

## Cell 13: CNN Markdown

Markdown explaining CNNs.

**What this does**: Explanatory text. No code.

---

## Cell 14: Define CNN Model

This is a big one! Let's break down the CNN architecture.

```python
class SimpleCNN(nn.Module):
```
- **What it does**: Defines a new class called SimpleCNN
  - `nn.Module`: Inherits from PyTorch's Module class (all neural networks do this)

```python
    """
    A simple CNN with 3 convolutional blocks

    Architecture:
    Block 1: Conv -> ReLU -> Pool (learns simple edges)
    Block 2: Conv -> ReLU -> Pool (learns shapes)
    Block 3: Conv -> ReLU -> Pool (learns objects)
    Classifier: Dropout -> Linear (makes final decision)
    """
```
- **What it does**: Documentation explaining the architecture

```python
    def __init__(self):
        super(SimpleCNN, self).__init__()
```
- **What it does**: Constructor method (runs when you create a new SimpleCNN)
  - `super().__init__()`: Calls the parent class constructor

```python
        self.features = nn.Sequential(
```
- **What it does**: Starts defining a sequential container for feature extraction layers

```python
            # Block 1: Detect simple patterns (edges, colors)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
```
- **What it does**: First convolutional layer
  - `3`: Input channels (RGB)
  - `32`: Output channels (32 different filters)
  - `kernel_size=3`: 3×3 filter
  - `padding=1`: Add 1 pixel border to keep size the same

```python
            nn.ReLU(),
```
- **What it does**: Applies ReLU activation (turns negative values to 0)

```python
            nn.MaxPool2d(2),
```
- **What it does**: Max pooling with 2×2 window
  - Reduces size from 32×32 to 16×16

```python
            # Block 2: Detect complex patterns (shapes, textures)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
```
- **What it does**: Second convolutional layer
  - `32`: Input channels (from previous layer)
  - `64`: Output channels (64 filters)

```python
            nn.ReLU(),
            nn.MaxPool2d(2),
```
- **What it does**: ReLU + Max pooling
  - Reduces size from 16×16 to 8×8

```python
            # Block 3: Detect high-level patterns (object parts)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
```
- **What it does**: Third convolutional layer
  - `64`: Input channels
  - `128`: Output channels (128 filters)

```python
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
```
- **What it does**: ReLU + Adaptive average pooling
  - `(1, 1)`: Output size
  - Reduces from 8×8 to 1×1 (global pooling)

```python
        )
```
- **What it does**: Closes the Sequential container

```python
        self.classifier = nn.Sequential(
```
- **What it does**: Starts defining the classification layers

```python
            nn.Dropout(0.5),
```
- **What it does**: Dropout layer
  - `0.5`: Randomly turns off 50% of neurons during training
  - Prevents overfitting

```python
            nn.Linear(128, 10)
```
- **What it does**: Fully connected layer
  - `128`: Input features (from previous layer)
  - `10`: Output classes (airplane, car, bird, etc.)

```python
        )
```
- **What it does**: Closes the Sequential container

```python
    def forward(self, x):
        """Pass input through the network"""
```
- **What it does**: Defines the forward pass (how data flows through the network)

```python
        x = self.features(x)
```
- **What it does**: Passes input through feature extraction layers

```python
        x = x.view(x.size(0), -1)
```
- **What it does**: Flattens the tensor
  - `x.size(0)`: Batch size
  - `-1`: Automatically calculate the remaining dimension
  - Example: (32, 128, 1, 1) → (32, 128)

```python
        x = self.classifier(x)
```
- **What it does**: Passes through classification layers

```python
        return x
```
- **What it does**: Returns the output (10 class scores)

```python
model = SimpleCNN().to(device)
```
- **What it does**: Creates an instance of SimpleCNN and moves it to the device (GPU or CPU)

```python
print("Model created!")
print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters()):,}")
print("These are the numbers the model will learn during training!")
```
- **What it does**: Prints model information
  - `p.numel()`: Number of elements in parameter p
  - `sum(...)`: Total number of parameters
  - `:,`: Formats with commas (94,538 instead of 94538)

---

## Cell 15: Prepare Data Markdown

Markdown explaining we'll create data loaders.

**What this does**: Just text. No code.

---

## Cell 16: Create Data Loaders

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```
- **What it does**: Defines transformations for training data
  - `RandomHorizontalFlip()`: 50% chance to flip
  - `RandomCrop(32, padding=4)`: Pad by 4 pixels, then randomly crop to 32×32
  - `ToTensor()`: Convert to tensor
  - `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`: Normalize each channel to mean=0.5, std=0.5

```python
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```
- **What it does**: Defines transformations for test data
  - NO augmentation! Only ToTensor and Normalize

```python
train_dataset_aug = datasets.CIFAR10(root='./data', train=True, transform=train_transform)
test_dataset_norm = datasets.CIFAR10(root='./data', train=False, transform=test_transform)
```
- **What it does**: Loads datasets with the new transformations

```python
train_loader = DataLoader(train_dataset_aug, batch_size=128, shuffle=True, num_workers=2)
```
- **What it does**: Creates a data loader for training
  - `batch_size=128`: Load 128 images at a time
  - `shuffle=True`: Randomly shuffle the data each epoch
  - `num_workers=2`: Use 2 parallel workers to load data

```python
test_loader = DataLoader(test_dataset_norm, batch_size=256, shuffle=False, num_workers=2)
```
- **What it does**: Creates a data loader for testing
  - `batch_size=256`: Larger batch size (no backprop, so more memory available)
  - `shuffle=False`: Don't shuffle test data

```python
print("Data ready!")
```
- **What it does**: Confirms everything is set up

---

## Cell 17: Training Markdown

Markdown explaining the training process.

**What this does**: Just text. No code.

---

## Cell 18: Train the Model

This is the heart of the notebook! Let's break it down.

```python
criterion = nn.CrossEntropyLoss()
```
- **What it does**: Defines the loss function
  - CrossEntropyLoss: Measures how wrong predictions are
  - Good for classification problems

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- **What it does**: Defines the optimizer
  - `Adam`: Adaptive optimization algorithm
  - `model.parameters()`: The weights to optimize
  - `lr=0.001`: Learning rate (how big each update is)

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
```
- **What it does**: Defines a function to train for one epoch

```python
    model.train()
```
- **What it does**: Sets model to training mode
  - Enables dropout, batch normalization, etc.

```python
    total, correct = 0, 0
```
- **What it does**: Initializes counters for accuracy calculation

```python
    for images, labels in tqdm(loader, desc="Training"):
```
- **What it does**: Loops through batches with a progress bar
  - `images`: Batch of images (128, 3, 32, 32)
  - `labels`: Batch of labels (128,)

```python
        images, labels = images.to(device), labels.to(device)
```
- **What it does**: Moves data to the device (GPU or CPU)

```python
        # Forward pass
        outputs = model(images)
```
- **What it does**: Passes images through the model
  - `outputs`: Shape (128, 10) - 10 scores for each image

```python
        loss = criterion(outputs, labels)
```
- **What it does**: Calculates the loss (how wrong the predictions are)

```python
        # Backward pass
        optimizer.zero_grad()
```
- **What it does**: Resets gradients to zero
  - **Important**: Gradients accumulate by default, so we must reset

```python
        loss.backward()
```
- **What it does**: Calculates gradients (how to adjust weights)
  - This is backpropagation!

```python
        optimizer.step()
```
- **What it does**: Updates the weights based on gradients
  - This is where learning happens!

```python
        # Track accuracy
        _, predicted = outputs.max(1)
```
- **What it does**: Gets the predicted class
  - `outputs.max(1)`: Returns (max_values, indices)
  - `_`: We don't care about max values
  - `predicted`: The indices (predicted classes)

```python
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
```
- **What it does**: Updates accuracy counters
  - `labels.size(0)`: Number of images in this batch
  - `predicted.eq(labels)`: True where prediction matches label
  - `.sum().item()`: Count how many were correct

```python
    return 100. * correct / total
```
- **What it does**: Returns accuracy as a percentage

```python
def test(model, loader, device):
    """Evaluate the model"""
```
- **What it does**: Defines a function to evaluate the model

```python
    model.eval()
```
- **What it does**: Sets model to evaluation mode
  - Disables dropout, etc.

```python
    total, correct = 0, 0

    with torch.no_grad():
```
- **What it does**: Disables gradient calculation
  - Saves memory and speeds up computation
  - We don't need gradients for evaluation

```python
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
```
- **What it does**: Same as training, but without gradient computation

```python
    return 100. * correct / total
```
- **What it does**: Returns test accuracy

```python
print("Starting training...\n")
for epoch in range(1, 6):
```
- **What it does**: Trains for 5 epochs (epochs 1 through 5)

```python
    print(f"Epoch {epoch}/5")
    train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_acc = test(model, test_loader, device)
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy:  {test_acc:.2f}%\n")
```
- **What it does**: For each epoch:
  1. Trains for one epoch
  2. Tests on test set
  3. Prints both accuracies

```python
print("\n" + "="*50)
print("FINAL CNN RESULTS")
print("="*50)
print(f"Final Test Accuracy: {test_acc:.2f}%")
print("="*50)
print("\n💡 Much better than feature engineering (~40%)!")
print("   CNNs learn the best features automatically!")
```
- **What it does**: Prints final results and explanation

---

## Cell 19: Summary Markdown

Final markdown cell summarizing what we learned.

**What this does**: Provides a summary and tips for the competition. Just text.

---

## Summary: Key Python Concepts Used

If you're new to Python, here are the key concepts used in this notebook:

### 1. Importing Libraries
```python
import library_name
from library_name import specific_function
```

### 2. Lists
```python
my_list = [1, 2, 3]
my_list.append(4)  # Add to end
```

### 3. Loops
```python
for i in range(10):  # Loop 10 times
    print(i)

for item in my_list:  # Loop through list
    print(item)
```

### 4. Functions
```python
def my_function(argument1, argument2=default_value):
    result = argument1 + argument2
    return result
```

### 5. Classes
```python
class MyClass:
    def __init__(self):  # Constructor
        self.variable = 0

    def method(self):  # Method
        return self.variable
```

### 6. List Comprehensions
```python
squares = [x**2 for x in range(10)]
# Creates: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### 7. String Formatting
```python
name = "Alice"
age = 25
print(f"My name is {name} and I'm {age} years old")
# Output: My name is Alice and I'm 25 years old
```

### 8. Indexing and Slicing
```python
my_list = [0, 1, 2, 3, 4]
my_list[0]    # First element: 0
my_list[-1]   # Last element: 4
my_list[1:3]  # Slice: [1, 2]
```

### 9. NumPy Arrays
```python
import numpy as np
arr = np.array([1, 2, 3])
arr.shape  # Dimensions
arr.mean() # Average
```

### 10. PyTorch Tensors
```python
import torch
tensor = torch.tensor([1, 2, 3])
tensor.to(device)  # Move to GPU/CPU
tensor.shape       # Dimensions
```

---

## Next Steps

Now that you understand the code:

1. **Run the notebook yourself**: Execute each cell and see the results
2. **Experiment**: Try changing parameters and see what happens
3. **Read the comprehensive review**: See `COMPREHENSIVE_REVIEW.md` for concept explanations
4. **Start the competition**: Modify the code for CIFAR-100

**Good luck! 🚀**
