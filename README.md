# ACM AI Workshop - Computer Vision

Welcome to the ACM AI Workshop on Computer Vision and Image Classification!

Kaggle Competition Link: https://www.kaggle.com/t/5822d84800d7461ea402e108d8d84e61

## 📑 Table of Contents
- [Documentation](#-documentation)
- [Workshop Structure](#-workshop-structure)
- [Getting Started](#-getting-started)
- [Competition Workflow](#-competition-workflow)
- [Tips for Success](#-tips-for-success)
- [Expected Performance](#-expected-performance)
- [Troubleshooting](#-troubleshooting)
- [Additional Resources](#-additional-resources)
- [Getting Help](#-getting-help)
- [Repository Contents](#-repository-contents)

---

## 📖 Documentation

New to the workshop or need to review? Check out our comprehensive guides:

- **[Setup Guide](SETUP_GUIDE.md)** - Install VSCode, Git, Python, and all dependencies
- **[Comprehensive Review](COMPREHENSIVE_REVIEW.md)** - Complete concept explanations from the workshop
- **[Code Walkthrough](CODE_WALKTHROUGH.md)** - Line-by-line explanation of the demo notebook

## 📚 Workshop Structure

### Part 1: Demo - CIFAR-10 (Instructor-led)
**File:** `cifar10_demo_student.ipynb`

In this interactive demo, you'll learn:
- What image classification is
- Feature engineering (the traditional way)
- Data augmentation (making models robust)
- Convolutional Neural Networks (the modern way)

**👉 Follow along as the instructor walks through the notebook!**

**Need help understanding the code?** See the [Code Walkthrough](CODE_WALKTHROUGH.md) for detailed line-by-line explanations.

---

### Part 2: Competition - CIFAR-10 Classification (Hands-on)
**Folder:** `cifar10_comp/`

Build a CNN to classify CIFAR-10 images (10 classes) and compete on Kaggle!

**Goal:** Achieve the highest accuracy on the augmented test set.

---

## 🚀 Getting Started

### Prerequisites

**First time setting up?** Follow our [Setup Guide](SETUP_GUIDE.md) to install:
- VSCode (code editor)
- Git (version control)
- Python 3.15+ (programming language)
- All required packages

### Quick Start (If you already have Python and Git)

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install torch torchvision pandas pillow tqdm matplotlib scikit-learn scikit-image
```

**Having installation issues?** Check the [Troubleshooting section](SETUP_GUIDE.md#troubleshooting) in the Setup Guide.

#### 3. Open in VSCode

```bash
code .
```

Or manually: File → Open Folder in VSCode

---

### Workshop Sessions

#### Demo Session (Follow Along)

Open the demo notebook:
```bash
jupyter notebook cifar10_demo_student.ipynb
```

The instructor will guide you through completing the code!

**Confused about what's happening?**
- During the workshop: Ask questions!
- After the workshop: Read the [Code Walkthrough](CODE_WALKTHROUGH.md) and [Comprehensive Review](COMPREHENSIVE_REVIEW.md)

#### Competition Session (Hands-on)

Navigate to the competition folder:
```bash
cd cifar10_comp
```

You have two options for working on the competition:

#### Option A: Jupyter Notebook (Recommended for beginners)
```bash
jupyter notebook cifar10_comp.ipynb
```

This notebook includes:
- Data exploration and visualization
- Starter CNN code to modify
- Interactive training cells

#### Option B: Python Scripts (For advanced users)
```bash
# Train your model
python main.py --epochs 20

# Generate submission from trained model
python kaggle_submission.py
```

---

## 📊 Competition Workflow

### Step 1: Download Test Data from Kaggle

Download these files from the Kaggle competition page:
- `test.csv` - List of test image IDs
- `test_images.zip` - Test images folder

Place both files in the `cifar10_comp/` folder, then unzip:
```bash
cd cifar10_comp
unzip test_images.zip
```

### Step 2: (Optional) Explore the Data

Open `cifar10_comp.ipynb` and run the data exploration cells to understand:
- What CIFAR-10 classes look like
- Class distribution
- Image statistics

**Note:** This is helpful for understanding the data but not required for training.

### Step 3: Improve Your Model

Modify `model.py` to improve the CNN architecture:
- Add more convolutional layers
- Add BatchNorm layers
- Try different architectures
- Experiment with dropout rates

**Key Tips:**
- The baseline `SimpleCNN` achieves ~50-60% accuracy
- Adding BatchNorm and more layers can boost to 70%+

### Step 4: Add Data Augmentation

Modify the `get_transforms()` function in `main.py` to add augmentations:

```python
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
transforms.RandomRotation(15),
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
transforms.RandomGrayscale(p=0.1),
```

**This is CRITICAL!** The test set has augmentations (noise, blur, color shifts).

### Step 5: Train Your Model and Generate Submission

Run the training script:
```bash
python main.py --epochs 20 --lr 0.001 --batch_size 128
```

This will:
1. Train your model on CIFAR-10
2. Save the best model as `best_model.pth`
3. **Automatically generate predictions** on the test set
4. Create `submission.csv` with your predictions

**That's it!** The script does everything for you.

### Step 6: Submit to Kaggle

Upload `submission.csv` to Kaggle and check your score on the leaderboard!

---

### Alternative: Manual Submission Generation

If you need to regenerate predictions without retraining, you can use:

```bash
python kaggle_submission.py
```

**Note:** You'll need to modify `kaggle_submission.py` to use your model architecture if you changed it from the baseline `SimpleCNN`.

---

## 💡 Tips for Success

### 1. Data Augmentation is KEY! 🔑

The test set has heavy augmentations (noise, blur, color shifts, rotations).

**Add augmentations to your training data** to generalize better:
```python
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
transforms.RandomRotation(15),
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
transforms.RandomGrayscale(p=0.1),
transforms.GaussianBlur(kernel_size=3)
```

**Impact:** Can improve score by 10-15%!

### 2. Improve the Model Architecture

The baseline `SimpleCNN` is simple. Try:
- Adding more convolutional blocks (4-6 blocks total)
- Using BatchNorm after Conv layers: `nn.BatchNorm2d(channels)`
- Trying deeper networks (more filters: 256, 512)
- Adding residual connections (ResNet-style)

**Impact:** Can improve score by 5-10%

### 3. Train Longer

The default is 10 epochs. Try training for 20-30 epochs!
- Monitor validation accuracy to avoid overfitting
- Use early stopping if accuracy plateaus

**Impact:** Can improve score by 2-5%

### 4. Experiment with Hyperparameters

Try different values:
- **Learning rate:** 0.0001, 0.001, 0.01
- **Batch size:** 64, 128, 256
- **Optimizer:** Adam, SGD with momentum, AdamW
- **Dropout rate:** 0.3, 0.5, 0.7

**Impact:** Can improve score by 1-3%

### 5. Monitor Overfitting

Watch for signs of overfitting:
- Training accuracy >> Test accuracy
- Test accuracy stops improving

**Solutions:**
- Add more data augmentation
- Increase dropout
- Use weight decay (AdamW optimizer)
- Train for fewer epochs

---

## 🏆 Submission Format

Your `submission.csv` must have this format:
```csv
id,label
0,3
1,8
2,5
...
```

- **id:** Image ID (from test.csv)
- **label:** Predicted class (0-9 for CIFAR-10)

The submission generation scripts handle this automatically!

---

## 🐛 Troubleshooting

### "test.csv not found"
Download `test.csv` from Kaggle and place it in the `cifar10_comp/` folder.

### "test_images/ not found"
Download and unzip `test_images.zip` from Kaggle into the `cifar10_comp/` folder.

### "CUDA out of memory"
Reduce batch size:
- In notebook: Change `BATCH_SIZE = 64` (or 32)
- In script: `python main.py --batch_size 64`

### Low Kaggle score despite good training accuracy?
You're overfitting to clean images! **Add more data augmentation.**

### ImportError: No module named 'torch'
Install dependencies: `pip install -r requirements.txt`

### Model not improving after a few epochs?
- Try a different learning rate
- Add data augmentation
- Make sure BatchNorm is added to your model
- Train longer (20-30 epochs)

---

## 🎯 Expected Performance

| Configuration | Validation Acc | Expected Kaggle Score |
|--------------|----------------|----------------------|
| Baseline (SimpleCNN, no aug) | ~55% | ~50-55% |
| Baseline + augmentation | ~65% | ~60-65% |
| Improved model + augmentation | ~70-75% | ~68-73% |
| Advanced techniques | ~75-80% | ~73-78% |

*Note: Kaggle test scores are typically 2-5% lower than validation due to augmentations*

---

## 📚 Additional Resources

### Workshop Documentation
- **[Setup Guide](SETUP_GUIDE.md)** - Complete setup instructions for VSCode, Git, Python, and packages
- **[Comprehensive Review](COMPREHENSIVE_REVIEW.md)** - Detailed explanations of all concepts covered
- **[Code Walkthrough](CODE_WALKTHROUGH.md)** - Line-by-line breakdown of the demo notebook

### External Resources
- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **CIFAR-10 Dataset:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Data Augmentation:** https://pytorch.org/vision/stable/transforms.html
- **CNN Architectures:** https://pytorch.org/vision/stable/models.html
- **Fast.ai Course:** https://www.fast.ai/ (practical deep learning)
- **Stanford CS231n:** http://cs231n.stanford.edu/ (computer vision course)

---

## 🤝 Getting Help

### During the Workshop
- Ask questions during the workshop!
- Check the Kaggle competition discussion forum
- Work with your neighbors

### After the Workshop
- **Read the guides:** [Comprehensive Review](COMPREHENSIVE_REVIEW.md) explains all concepts
- **Understand the code:** [Code Walkthrough](CODE_WALKTHROUGH.md) breaks down every line
- **Setup issues?** [Setup Guide](SETUP_GUIDE.md) has troubleshooting tips

---

## 🎓 Learning Objectives

By the end of this workshop, you should be able to:

✅ Understand how CNNs work for image classification
✅ Build and train a CNN using PyTorch
✅ Apply data augmentation to improve robustness
✅ Tune hyperparameters for better performance
✅ Generate predictions and submit to Kaggle
✅ Debug common issues in model training

**Don't understand everything yet?** That's normal! Review the [Comprehensive Review](COMPREHENSIVE_REVIEW.md) at your own pace.

---

## 📁 Repository Contents

```
workshop/
├── README.md                           # You are here!
├── SETUP_GUIDE.md                      # Installation and setup instructions
├── COMPREHENSIVE_REVIEW.md             # Detailed concept explanations
├── CODE_WALKTHROUGH.md                 # Line-by-line code breakdown
├── requirements.txt                    # Python dependencies
├── cifar10_demo_student.ipynb          # Demo notebook (follow along)
│
└── cifar10_comp/                       # Competition folder
    ├── cifar10_comp.ipynb              # Competition notebook (interactive)
    ├── model.py                        # CNN architecture (modify this!)
    ├── main.py                         # Training script (modify this!)
    ├── kaggle_submission.py            # Generate submission
    │
    ├── best_model.pth                  # Trained model (generated after training)
    ├── submission.csv                  # Predictions (generated before submission)
    │
    ├── test.csv                        # Test IDs (download from Kaggle)
    └── test_images/                    # Test images (download from Kaggle)
```

---

Good luck and have fun! 🚀

**Remember:** The key to winning is **data augmentation** + **good architecture** + **proper training**!

**Feeling lost?** Start with the [Setup Guide](SETUP_GUIDE.md), then read the [Comprehensive Review](COMPREHENSIVE_REVIEW.md) to understand the concepts!
