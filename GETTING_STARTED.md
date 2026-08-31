# CHANA: Cell Histology Automated Neural Network Analyzer
## End-to-End User Setup & Quick Start Guide

This guide walks you through setting up your environment, downloading the neural network models, and running automated osteoclast segmentation and cell counting on your computer.

---

## 📋 Table of Contents
1. [Prerequisites & Initial Software Installation](#1-prerequisites--initial-software-installation)
   * [macOS (Apple Silicon & Intel)](#macos-apple-silicon--intel)
   * [Windows (10/11)](#windows-1011)
2. [Recommended Workspace & Folder Organization](#2-recommended-workspace--folder-organization)
3. [Cloning the Repository & Downloading Model Weights](#3-cloning-the-repository--downloading-model-weights)
4. [Setting Up the Python Environment](#4-setting-up-the-python-environment)
5. [Validating Checkpoints (Sanity Check)](#5-validating-checkpoints-sanity-check)
6. [Testing on Sample / Practice Data](#6-testing-on-sample--practice-data)
7. [Running CHANA on Your Own Images](#7-running-chana-on-your-own-images)
   * [Single Image Inference](#single-image-inference)
   * [Batch Processing Multiple Images](#batch-processing-multiple-images)
8. [Parameter Tuning & Available Checkpoints](#8-parameter-tuning--available-checkpoints)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. Prerequisites & Initial Software Installation

Before running CHANA, ensure you have the following core software installed:

### macOS (Apple Silicon & Intel)

1. **Visual Studio Code (VS Code):**
   * Download and install from [code.visualstudio.com](https://code.visualstudio.com/).
   * Install the **Python Extension** inside VS Code (Search for `Python` in the Extensions sidebar `Cmd+Shift+X`).

2. **Git & Git LFS (Large File Storage):**
   * *Why:* CHANA stores neural network model weights (~2.5 GB) using Git LFS. Without Git LFS, checkpoint files will only download as unreadable 134-byte pointer files.
   * **Install via Homebrew:**
     ```bash
     brew install git-lfs
     git lfs install
     ```
   * *Or manual download:* Download the `.zip` from [git-lfs.github.com](https://git-lfs.github.com/) (select **`darwin-arm64`** for M1/M2/M3 Macs or **`darwin-amd64`** for Intel Macs), unzip, and run `sudo ./install.sh`.

3. **Conda (Miniforge / Miniconda):**
   * CHANA runs on **Python 3.10** with **TensorFlow 2.16.2**.
   * Download and install [Miniforge (Apple Silicon)](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   * Initialize conda for your terminal:
     ```bash
     conda init zsh
     ```

---

### Windows (10/11)

1. **Visual Studio Code (VS Code):**
   * Download and run the Windows installer from [code.visualstudio.com](https://code.visualstudio.com/).
   * Open VS Code and install the official **Python Extension** from Microsoft.

2. **Git for Windows (Includes Git LFS):**
   * Download from [gitforwindows.org](https://gitforwindows.org/).
   * During installation, ensure the **"Git LFS (Large File Support)"** checkbox is selected.
   * Open **Git Bash** or **PowerShell** and run:
     ```powershell
     git lfs install
     ```

3. **Conda (Miniconda / Miniforge):**
   * Download the 64-bit Windows installer from [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   * Run the installer and check **"Register Miniconda as my default Python"**.

---

## 2. Recommended Workspace & Folder Organization

To ensure paths and scripts resolve seamlessly, we recommend maintaining a dedicated project directory (e.g. `Projects/` or `Documents/CHANA/`):

```
CHANA/                           <-- Open THIS folder as your root workspace in VS Code
├── configs/                     # Model and dataset configuration YAMLs
├── environment/
│   └── inference.yml            # Environment specification (Python 3.10, TF 2.16.2)
├── inputs/                      # <-- Create this folder for your microscopy images (.tif, .png)
├── manifests/                   # Model registry and SHA-256 hash manifests
├── models/                      # Pretrained neural network weights (.weights.h5)
├── outputs/                     # <-- Auto-generated masks, labels, and measurement CSVs
├── sample_data/
│   └── public_example/          # Practice image, reference masks, and evaluation pairs
├── scripts/
│   ├── predict.py               # Single-image inference & cell counting
│   ├── compare_models.py        # Model benchmarking & validation metrics
│   └── validate_checkpoints.py  # Checkpoint integrity validator
└── src/                         # Core CHANA Python package
```

> **Important:** Always open the root `CHANA` folder in VS Code (`File > Open Folder... > select CHANA`). All scripts use relative paths based from the repository root.

---

## 3. Cloning the Repository & Downloading Model Weights

Open your terminal (macOS) or PowerShell / Git Bash (Windows) and clone the repository with Git LFS enabled:

```bash
# 1. Initialize Git LFS on your machine
git lfs install

# 2. Clone the repository
git clone https://github.com/yuvipaloozie/CHANA.git

# 3. Navigate into the repository
cd CHANA

# 4. Pull all model checkpoint files (~2.5 GB)
git lfs pull
```

---

## 4. Setting Up the Python Environment

CHANA provides a pre-configured environment file (`environment/inference.yml`) that locks all core dependencies (TensorFlow 2.16.2, Keras 3.12.4, OpenCV, scikit-image, scipy, pandas).

```bash
# 1. Create the isolated Conda environment
conda env create -f environment/inference.yml

# 2. Activate the environment
conda activate chana-inference

# 3. Install CHANA as an editable package
python -m pip install -e .
```

> **Tip for VS Code users:** Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac), type **`Python: Select Interpreter`**, and choose **`Python 3.10.x ('chana-inference': conda)`**.

---

## 5. Validating Checkpoints (Sanity Check)

Before processing images, verify that all model weights downloaded completely and match their SHA-256 cryptographic checksums:

```bash
python scripts/validate_checkpoints.py --weights-dir models --hash-only
```

You should see all 6 model checkpoints validated with green confirmation marks.

---

## 6. Testing on Sample / Practice Data

Run the complete pipeline on the included public bright-field osteoclast sample using the top-performing **U-Net++ Curriculum** model:

```bash
python scripts/predict.py \
  --input sample_data/public_example/input_image.tif \
  --output-dir outputs/sample_test \
  --model-id unetpp_curriculum \
  --weights-dir models
```

### Generated Outputs in `outputs/sample_test/`:
* **`input_image_mask.png`**: Binary segmentation mask (white = osteoclasts, black = background).
* **`input_image_objects.csv`**: Morphometry table detailing individual cell statistics (Area in \(px^2\), centroid \(x, y\), perimeter, circularity, eccentricity).
* **`input_image_labels.tif`**: 16-bit watershed label map separating adjacent/touching osteoclasts.
* **`input_image_probability.npy`**: Raw floating-point probability map.

---

## 7. Running CHANA on Your Own Images

### Single Image Inference

1. Place your `.tif`, `.png`, or `.jpg` image into the `inputs/` folder.
2. Run:
   ```bash
   python scripts/predict.py \
     --input inputs/my_culture_tile.tif \
     --output-dir outputs/my_culture_tile \
     --model-id unetpp_curriculum \
     --weights-dir models
   ```

---

### Batch Processing Multiple Images

To automatically count osteoclasts and generate morphometry tables across multiple images:

#### On macOS / Linux (Terminal):
```bash
for img in inputs/*.tif; do
  name=$(basename "$img" .tif)
  echo "--- Processing: $name ---"
  python scripts/predict.py \
    --input "$img" \
    --output-dir "outputs/$name" \
    --model-id unetpp_curriculum \
    --weights-dir models
done
```

#### On Windows (PowerShell):
```powershell
Get-ChildItem -Path "inputs\*.tif" | ForEach-Object {
    $name = $_.BaseName
    Write-Host "--- Processing: $name ---"
    python scripts\predict.py `
      --input $_.FullName `
      --output-dir "outputs\$name" `
      --model-id unetpp_curriculum `
      --weights-dir models
}
```

#### Summarize All Detected Cell Counts:
After running batch inference, print a consolidated count list with:
```bash
for f in outputs/*/*_objects.csv; do
  img_name=$(basename "$(dirname "$f")")
  num=$(tail -n +2 "$f" | wc -l | tr -d ' ')
  echo "$img_name: $num osteoclasts detected"
done
```

---

## 8. Parameter Tuning & Available Checkpoints

### Available Models (`--model-id`):
| Model ID | Architecture | Training Regime | Recommendation |
| :--- | :--- | :--- | :--- |
| **`unetpp_curriculum`** | U-Net++ (ResNet-50) | 4-Stage Domain Curriculum | **Recommended (Best Accuracy & F1: 0.842)** |
| `unetpp_baseline` | U-Net++ (ResNet-50) | Expert-Real Only | Baseline comparator |
| `unet_curriculum` | U-Net (DenseNet-121) | 4-Stage Domain Curriculum | Strong generalizer |
| `unet_baseline` | U-Net (DenseNet-121) | Expert-Real Only | Baseline comparator |
| `transunet_curriculum` | TransUNet | 4-Stage Domain Curriculum | Transformer hybrid |
| `transunet_baseline` | TransUNet | Expert-Real Only | Baseline comparator |

### Common CLI Options:
* `--threshold <float>`: Foreground probability threshold (Default: `0.5`).
* `--min-area <float>`: Minimum osteoclast area in pixels (Default: `50.0`). Filters out debris/mononuclear cells.
* `--max-area <float>`: Maximum osteoclast area in pixels (Default: `10000.0`).
* `--min-peak-distance <int>`: Watershed separation sensitivity (Default: `20`). Increase for larger cells, decrease for dense/small clusters.

---

## 9. Troubleshooting & FAQ

### Q: `validate_checkpoints.py` reports hash mismatches or files are only ~134 bytes.
* **Cause:** Git LFS was not installed or pulled prior to running.
* **Solution:** Run `git lfs install` followed by `git lfs pull` from the root `CHANA` folder.

### Q: `ModuleNotFoundError: No module named 'chana'`
* **Cause:** The CHANA package was not installed in editable mode.
* **Solution:** Run `python -m pip install -e .` from within the root `CHANA` directory with your `chana-inference` conda environment active.

### Q: `No matching distribution found for tensorflow==2.16.2`
* **Cause:** Your active Python environment is Python 3.12+ or 3.14+.
* **Solution:** TensorFlow 2.16.2 requires **Python 3.10**. Re-create the environment using `conda env create -f environment/inference.yml`.

---

## 📜 Citation
If you use CHANA in your research, please cite:
```bibtex
@software{chana2026,
  title = {CHANA: Cell Histology Automated Neural Network Analyzer},
  author = {Szabo, Sarah and collaborators},
  year = {2026},
  url = {https://github.com/yuvipaloozie/CHANA}
}
```
