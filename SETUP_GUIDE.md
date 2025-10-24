# Setup Guide - VSCode and Git Installation

This guide will help you set up your development environment for the Computer Vision workshop and competition.

---

## Table of Contents
1. [VSCode Installation](#vscode-installation)
   - [macOS](#vscode-on-macos)
   - [Windows](#vscode-on-windows)
2. [Git Installation](#git-installation)
   - [macOS](#git-on-macos)
   - [Windows](#git-on-windows)
3. [Python Installation](#python-installation)
4. [Setting Up the Project](#setting-up-the-project)
5. [Installing Python Packages](#installing-python-packages)
6. [Troubleshooting](#troubleshooting)

---

## VSCode Installation

Visual Studio Code (VSCode) is a free, powerful code editor. It's perfect for Python development and has great Jupyter Notebook support.

### VSCode on macOS

#### Method 1: Download from Website (Recommended for Beginners)

1. **Go to the VSCode website**:
   - Open your browser and visit: https://code.visualstudio.com/

2. **Download VSCode**:
   - Click the big blue "Download for Mac" button
   - The file `VSCode-darwin-universal.zip` will download

3. **Install VSCode**:
   - Open your Downloads folder
   - Double-click the downloaded zip file to extract it
   - Drag the "Visual Studio Code" app to your Applications folder
   - Open VSCode from your Applications folder

4. **First Launch**:
   - When you first open VSCode, macOS might show a security warning
   - Click "Open" to confirm you want to run it

#### Method 2: Install via Homebrew (For Advanced Users)

If you have Homebrew installed:
```bash
brew install --cask visual-studio-code
```

### VSCode on Windows

#### Step-by-Step Installation:

1. **Download VSCode**:
   - Go to https://code.visualstudio.com/
   - Click "Download for Windows"
   - The file `VSCodeUserSetup-x64-*.exe` will download

2. **Run the Installer**:
   - Double-click the downloaded file
   - Click "Yes" if Windows asks for permission

3. **Installation Wizard**:
   - Click "Next" to accept the license agreement
   - Click "Next" to choose the default installation location
   - **Important**: On the "Select Additional Tasks" screen, CHECK these boxes:
     - ✅ Add "Open with Code" action to Windows Explorer file context menu
     - ✅ Add "Open with Code" action to Windows Explorer directory context menu
     - ✅ Add to PATH (this is usually checked by default)
   - Click "Next", then "Install"

4. **Finish**:
   - Click "Finish" to launch VSCode

---

## Git Installation

Git is a version control system that lets you track changes to your code and download projects from GitHub.

### Git on macOS

#### Installing Git via Xcode Command Line Tools (Recommended)

macOS doesn't come with Git pre-installed, but you can easily install it through Xcode Command Line Tools.

1. **Open Terminal**:
   - Press `Cmd + Space` to open Spotlight
   - Type "Terminal" and press Enter

2. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

3. **Confirm Installation**:
   - A popup window will appear asking "The xcode-select command requires the command line developer tools. Would you like to install the tools now?"
   - Click "Install"
   - Click "Agree" to the license agreement
   - Wait for the download and installation (this may take 10-20 minutes)

4. **Verify Git is Installed**:
   ```bash
   git --version
   ```
   - You should see something like: `git version 2.39.2 (Apple Git-143)`

#### Alternative: Install Git via Homebrew

If you have Homebrew installed:
```bash
brew install git
```

### Git on Windows

Good news! Git does **NOT** come pre-installed on Windows, so we need to install it.

#### Step-by-Step Installation:

1. **Download Git for Windows**:
   - Go to https://git-scm.com/download/win
   - The download should start automatically
   - If not, click "Click here to download manually"

2. **Run the Installer**:
   - Double-click the downloaded file `Git-*-64-bit.exe`
   - Click "Yes" if Windows asks for permission

3. **Installation Wizard** (Use These Settings):
   - **Select Destination**: Keep the default, click "Next"
   - **Select Components**: Keep defaults, click "Next"
   - **Select Start Menu Folder**: Keep default, click "Next"
   - **Choose Default Editor**: Select "Use Visual Studio Code as Git's default editor", click "Next"
   - **Adjusting PATH**: Select "Git from the command line and also from 3rd-party software", click "Next"
   - **HTTPS Transport**: Keep "Use the OpenSSL library", click "Next"
   - **Line Ending Conversions**: Keep "Checkout Windows-style, commit Unix-style", click "Next"
   - **Terminal Emulator**: Keep "Use MinTTY", click "Next"
   - **Default Git Pull**: Keep "Default (fast-forward or merge)", click "Next"
   - **Credential Helper**: Keep "Git Credential Manager", click "Next"
   - **Extra Options**: Keep defaults, click "Next"
   - **Experimental Options**: Don't check anything, click "Install"

4. **Finish**:
   - Uncheck "View Release Notes"
   - Click "Finish"

5. **Verify Git is Installed**:
   - Open Command Prompt (search for "cmd" in Start menu)
   - Type:
   ```bash
   git --version
   ```
   - You should see something like: `git version 2.43.0.windows.1`

---

## Python Installation

You'll need Python 3.15 or later for this workshop.

### Check if Python is Already Installed

#### macOS:
```bash
python3 --version
```

#### Windows:
```bash
python --version
```

If you see a version number 3.15 or higher (e.g., `Python 3.15.1`), you're good! You can skip to [Setting Up the Project](#setting-up-the-project).

---

### Python on macOS

#### Step-by-Step Installation from Official Website:

1. **Go to the Python website**:
   - Open your browser and visit: https://www.python.org/downloads/

2. **Download Python**:
   - Click the big yellow "Download Python 3.x.x" button
   - This downloads the latest stable version
   - The file will be named something like `python-3.15.1-macos11.pkg`

3. **Run the Installer**:
   - Open your Downloads folder
   - Double-click the `.pkg` file
   - Click "Continue" to start the installation

4. **Installation Steps**:
   - **Introduction**: Click "Continue"
   - **Read Me**: Click "Continue"
   - **License**: Click "Continue", then "Agree"
   - **Installation Type**: Keep the default location, click "Install"
   - **Enter your password** when prompted (your macOS login password)
   - Wait for installation to complete (takes 1-2 minutes)
   - Click "Close" when done

5. **Verify Installation**:
   - Open Terminal (`Cmd + Space`, type "Terminal")
   - Type:
   ```bash
   python3 --version
   ```
   - You should see something like: `Python 3.15.1`

6. **Verify pip is installed**:
   ```bash
   pip3 --version
   ```
   - You should see something like: `pip 24.3 from ...`

**Important Notes for macOS**:
- macOS comes with an old version of Python 2.7 (don't use it!)
- Always use `python3` and `pip3` (not `python` or `pip`)
- The installer automatically adds Python to your PATH

---

### Python on Windows

#### Step-by-Step Installation:

1. **Go to the Python website**:
   - Visit: https://www.python.org/downloads/

2. **Download Python**:
   - Click the big yellow "Download Python 3.x.x" button
   - Make sure it's version 3.15 or higher
   - The file will be named something like `python-3.15.1-amd64.exe`

3. **Run the Installer**:
   - Double-click the downloaded file
   - Click "Yes" if Windows asks for permission

4. **VERY IMPORTANT - Before clicking anything else**:
   - ✅ **CHECK the box that says "Add Python to PATH"** (at the bottom)
   - This is critical! If you forget this, Python won't work in Command Prompt

5. **Installation Options**:
   - Click "Install Now" (recommended for most users)
   - Wait for installation (takes 1-2 minutes)
   - Click "Close" when done

6. **Verify Installation**:
   - Open Command Prompt (search for "cmd" in Start menu)
   - Type:
   ```bash
   python --version
   ```
   - You should see something like: `Python 3.15.1`

7. **Verify pip is installed**:
   ```bash
   pip --version
   ```
   - You should see something like: `pip 24.3 from ...`

**Important Notes for Windows**:
- If you forgot to check "Add Python to PATH", you'll need to reinstall
- On Windows, use `python` and `pip` (not `python3` or `pip3`)
- Restart Command Prompt after installation for PATH changes to take effect

---

## Setting Up the Project

Now that you have VSCode and Git installed, let's set up the workshop project!

### Step 1: Clone the Repository

#### macOS:
1. Open Terminal
2. Navigate to where you want to save the project:
   ```bash
   cd ~/Documents
   ```
3. Clone the repository (replace with the actual repo URL):
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
4. Navigate into the project folder:
   ```bash
   cd your-repo-name
   ```

#### Windows:
1. Open Command Prompt or PowerShell
2. Navigate to where you want to save the project:
   ```bash
   cd C:\Users\YourUsername\Documents
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
4. Navigate into the project folder:
   ```bash
   cd your-repo-name
   ```

### Step 2: Open Project in VSCode

#### From Terminal/Command Prompt:
```bash
code .
```
(The `.` means "current directory")

#### Or Manually:
1. Open VSCode
2. Click "File" → "Open Folder"
3. Navigate to your project folder and select it

---

## Installing Python Packages

You'll need to install several Python packages for the workshop.

### Step 1: Install Required Packages

Create a file called `requirements.txt` with these contents:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
tqdm>=4.65.0
jupyter>=1.0.0
ipykernel>=6.25.0
```

Then install everything:
```bash
pip install -r requirements.txt
```

### Alternative: Install Packages Individually

If you don't want to use `requirements.txt`:

```bash
pip install torch torchvision numpy matplotlib scikit-learn scikit-image tqdm jupyter ipykernel
```

### Step 2: Set Up Jupyter in VSCode

1. Open VSCode
2. Install the "Jupyter" extension:
   - Click the Extensions icon in the sidebar (or press `Ctrl+Shift+X` / `Cmd+Shift+X`)
   - Search for "Jupyter"
   - Click "Install" on the official Jupyter extension by Microsoft

3. Install the "Python" extension (if not already installed):
   - Search for "Python"
   - Install the official Python extension by Microsoft

4. Open a `.ipynb` file in your project
5. VSCode will automatically detect your virtual environment
6. Select the kernel: Click "Select Kernel" in the top-right → Choose your `venv` environment

---

## Troubleshooting

### Common Issues:

#### 1. "python: command not found" (macOS)
**Solution**: Use `python3` instead of `python`:
```bash
python3 --version
python3 -m venv venv
```

#### 2. "pip: command not found"
**Solution**: Use `pip3` (macOS) or reinstall Python with "Add to PATH" checked (Windows):
```bash
# macOS
pip3 install --upgrade pip

# Windows
python -m pip install --upgrade pip
```

#### 3. "Permission Denied" when installing packages (macOS)
**Solution**: Use a virtual environment (recommended) or add `--user`:
```bash
pip3 install --user torch torchvision
```

#### 4. VSCode doesn't recognize the virtual environment
**Solution**:
1. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows)
2. Type "Python: Select Interpreter"
3. Choose the Python interpreter from your `venv` folder

#### 5. Git commands don't work in VSCode terminal (Windows)
**Solution**:
1. Close and reopen VSCode (Git PATH needs to be loaded)
2. Or restart your computer

#### 6. Jupyter kernel keeps dying
**Solution**: You might need to install PyTorch with different settings (especially for macOS with M1/M2 chips):
```bash
# For macOS with Apple Silicon
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Verifying Your Setup

Run this Python script to verify everything is installed correctly:

Create a file called `test_setup.py`:

```python
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} installed")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import torchvision
    print(f"✓ torchvision {torchvision.__version__} installed")
except ImportError:
    print("✗ torchvision not installed")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} installed")
except ImportError:
    print("✗ NumPy not installed")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__} installed")
except ImportError:
    print("✗ Matplotlib not installed")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__} installed")
except ImportError:
    print("✗ scikit-learn not installed")

try:
    import skimage
    print(f"✓ scikit-image {skimage.__version__} installed")
except ImportError:
    print("✗ scikit-image not installed")

try:
    import tqdm
    print(f"✓ tqdm {tqdm.__version__} installed")
except ImportError:
    print("✗ tqdm not installed")

print("\nSetup verification complete!")
```

Run it:
```bash
python test_setup.py
```

You should see checkmarks (✓) for all packages.

---

## Next Steps

Now that your environment is set up:

1. **Explore the demo notebook**: Open `cifar10_demo_student.ipynb` in VSCode
2. **Read the comprehensive review**: See `COMPREHENSIVE_REVIEW.md`
3. **Read the code walkthrough**: See `CODE_WALKTHROUGH.md`
4. **Start the competition**: Modify `cifar10_comp/model.py` and `cifar10_comp/main.py`

---

## Additional Resources

### VSCode Tips:
- **Command Palette**: `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows)
- **Toggle Terminal**: ``Ctrl+` `` (backtick)
- **Split Editor**: `Cmd+\` (macOS) or `Ctrl+\` (Windows)

### Git Basics:
```bash
# Clone a repository
git clone <url>

# Check status
git status

# Pull latest changes
git pull

# Create a branch
git checkout -b my-branch

# Add files
git add .

# Commit changes
git commit -m "Your message"

# Push changes
git push origin my-branch
```

### Python Virtual Environments:
```bash
# Create
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate
```

---

## Getting Help

If you're still stuck:
1. Check the [VSCode Documentation](https://code.visualstudio.com/docs)
2. Check the [Git Documentation](https://git-scm.com/doc)
3. Ask on the ACM Discord: acmurl.com/discord
4. Email: contact@acmucsd.org

---

## Using Conda (Alternative Package Manager)

### What is Conda?

**Conda** is an alternative to pip and virtual environments. It's a package manager and environment manager that's especially popular in data science and machine learning.

#### Conda vs pip + venv:

| Feature | pip + venv | Conda |
|---------|-----------|-------|
| **What it manages** | Python packages only | Python packages + Python itself + non-Python dependencies |
| **Environment isolation** | Yes (venv) | Yes (conda environments) |
| **Installing Python versions** | No (need to install separately) | Yes (conda can install different Python versions) |
| **Binary packages** | Sometimes needs compilation | Pre-compiled binaries (faster, easier) |
| **Disk space** | Smaller | Larger (includes more dependencies) |
| **Popular in** | General Python development | Data science, ML, scientific computing |

#### When to use Conda:

✅ **Use Conda if**:
- You work with data science/ML libraries frequently
- You need to switch between Python versions easily
- You want pre-compiled packages (no compilation headaches)
- You're on Windows and have trouble installing packages with pip
- You need packages with complex C/C++ dependencies

❌ **Stick with pip + venv if**:
- You want a lightweight setup
- You're doing general Python development (web apps, scripts)
- Disk space is limited
- You're comfortable with pip

#### Two Conda Distributions:

1. **Anaconda** (Full version):
   - Size: ~3-5 GB
   - Includes: Python + conda + 250+ pre-installed packages
   - Best for: Beginners, people who want everything ready to go
   - Download: https://www.anaconda.com/download

2. **Miniconda** (Minimal version):
   - Size: ~400 MB
   - Includes: Python + conda only (install packages as needed)
   - Best for: People who want to choose what to install
   - Download: https://docs.conda.io/en/latest/miniconda.html

**For this workshop, we recommend Anaconda** 

---

### Using Conda for This Project

Now that you have Conda installed, here's how to use it for the workshop:

#### Step 1: Create a Conda Environment

Instead of using `venv`, we'll create a conda environment:

```bash
# Create environment named "cv-workshop" with Python 3.15
conda create -n cv-workshop python=3.15

# You'll be asked to confirm, type 'y' and press Enter
```

**What this does**: Creates an isolated environment with Python 3.15 installed.

#### Step 2: Activate the Environment

**macOS/Linux**:
```bash
conda activate cv-workshop
```

**Windows**:
```bash
conda activate cv-workshop
```

Your prompt should now show `(cv-workshop)` instead of `(base)`.

#### Step 3: Install Packages with Conda

Conda can install most packages directly:

```bash
# Install PyTorch and related packages
conda install pytorch torchvision -c pytorch

# Install other packages
conda install numpy matplotlib scikit-learn scikit-image jupyter ipykernel tqdm
```

**Alternative - Using pip within conda**:
If a package isn't available via conda, you can still use pip:
```bash
# Make sure your conda environment is activated first!
pip install some-package-not-in-conda
```

#### Step 4: Verify Installation

Check that everything is installed:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torchvision; print('torchvision version:', torchvision.__version__)"
```

#### Step 5: Use Jupyter with Conda Environment

To use your conda environment in Jupyter notebooks in VSCode:

1. **Make sure your environment is activated**:
   ```bash
   conda activate cv-workshop
   ```

2. **Install ipykernel** (if not already installed):
   ```bash
   conda install ipykernel
   ```

3. **Register the environment as a Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name=cv-workshop
   ```

4. **In VSCode**:
   - Open a `.ipynb` file
   - Click "Select Kernel" in the top-right
   - Choose "cv-workshop" from the list

---

### Common Conda Commands

Here's a quick reference for working with conda:

#### Environment Management:
```bash
# List all environments
conda env list

# Create new environment
conda create -n myenv python=3.15

# Activate environment
conda activate myenv

# Deactivate environment
conda deactivate

# Delete environment
conda env remove -n myenv
```

#### Package Management:
```bash
# Install package
conda install package-name

# Install specific version
conda install package-name=1.2.3

# Install from a specific channel
conda install package-name -c channel-name

# Update package
conda update package-name

# List installed packages
conda list

# Remove package
conda remove package-name
```

#### Updating Conda:
```bash
# Update conda itself
conda update conda

# Update all packages in current environment
conda update --all
```

#### Exporting/Importing Environments:
```bash
# Export environment to file
conda env export > environment.yml

# Create environment from file
conda env create -f environment.yml
```

---

### Troubleshooting Conda

#### 1. "conda: command not found" (macOS)
**Solution**: Initialize conda and restart Terminal:
```bash
~/miniconda3/bin/conda init zsh
# Then close and reopen Terminal
```

#### 2. "conda: command not found" (Windows)
**Solution**: You didn't check "Add to PATH" during installation. Either:
- Reinstall and check the box
- Or use "Anaconda Prompt" from the Start menu instead of Command Prompt

#### 3. "Solving environment: failed"
**Solution**: Try installing packages one at a time, or use pip:
```bash
conda install pytorch torchvision -c pytorch
pip install scikit-image
```

#### 4. Conda is taking too much disk space
**Solution**: Clean up cached packages:
```bash
conda clean --all
```

#### 5. Wrong Python version in conda environment
**Solution**: Recreate the environment with specific Python version:
```bash
conda env remove -n cv-workshop
conda create -n cv-workshop python=3.15
```

#### 6. Can't use conda environment in VSCode Jupyter
**Solution**: Install ipykernel and register the kernel:
```bash
conda activate cv-workshop
conda install ipykernel
python -m ipykernel install --user --name=cv-workshop
```

---

### Additional Conda Resources

- **Official Conda Documentation**: https://docs.conda.io/
- **Conda Cheat Sheet**: https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
- **Managing Environments**: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- **Miniconda vs Anaconda**: https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda

---

**Good luck with the workshop! 🚀**
