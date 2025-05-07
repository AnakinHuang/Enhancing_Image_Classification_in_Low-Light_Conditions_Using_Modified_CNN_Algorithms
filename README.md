# Enhancing Image Classification in Low-Light Conditions Using Modified CNN Algorithms

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Overview

This project explores techniques to improve image classification accuracy in low-light environments using **modified Convolutional Neural Networks (CNNs)** and advanced preprocessing methods. Leveraging the **Exclusively Dark (ExDark) dataset**, we benchmarked traditional CNNs and enhanced architectures (ResNet50, VGG19) across binary, ternary, and multilabel tasks.

---

## 🖼️ Dataset

- **Source:** [Exclusively Dark Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- **Classes:** 12 object classes (e.g., Car, Bottle, Dog, etc.)
- **Volume:** ~7,363 images; ~23,700 object annotations
- **Annotation:** Largest object per image was prioritized for labeling (via custom script)

📥 **Download links:**
- Dataset: [Google Drive](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view?usp=sharing)
- Annotations: [GitHub](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Groundtruth)
- Trained models: [Google Drive](https://drive.google.com/drive/folders/125vwohSUWu_U0SKjGf_yUEYx9faGN7Bt?usp=sharing)

---

## 🔧 Methods

### Preprocessing & Augmentation

- **Annotation merging:** Custom tool (`ExDark_annotator.py`) to preprocess and merge annotations.
- **Image enhancement:**
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Histogram Equalization
    - Dynamic Range Adjustment
    - Gaussian Blur
- **Data augmentation:**
    - Resizing (512×512 ➔ 256×256)
    - Random vertical flips
    - Cropping with scale (0.8–1.0) & aspect ratio (3/4–4/3)

### Normalization

- **Standard:** ImageNet parameters
- **Custom:** Calculated RGB mean & std from ExDark dataset

---

## 📊 Experiments & Results

### 1️⃣ Binary Classification (Creature vs. Auto)

| Model           | Accuracy | Notable Results                        |
|-----------------|----------|----------------------------------------|
| Custom CNN      | 73.1%    | Underfitting with ~0.69 loss plateau   |
| ResNet50        | 94.2%    | Balanced precision & recall            |
| VGG19           | 91.9%    | Slight bias toward 'Creature' class    |

### 2️⃣ Ternary Classification (People, Cat, Dog)

| Model           | Accuracy | Notes                                  |
|-----------------|----------|----------------------------------------|
| Custom CNN      | 52.8%    | Bias toward 'People' class             |
| ResNet50        | 92.4%    | Strong performance across all classes  |
| VGG19           | 90.6%    | Well-balanced results                  |

### 3️⃣ Multilabel N-ary Classification (12 classes)

| Model           | Accuracy | Strengths                              |
|-----------------|----------|----------------------------------------|
| Custom CNN      | 8.2%     | Struggled with complex multi-labeling  |
| ResNet50        | 60.1%    | High precision, esp. for Boat & Dog    |
| VGG19           | 54.8%    | Competent, some class confusion        |

---

## ✅ Requirements

- Python 3.8+
- Dependencies: See `venv.txt`

Install via:

```bash
pip install -r venv.txt
```

Key libraries:
- PyTorch & TorchVision
- OpenCV
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn, scikit-image

---

## 🛠️ Usage

1️⃣ **Preprocess annotations:**

```bash
python ExDark_annotator.py
```

2️⃣ **Run experiments:**
- Open `csc245_proj3.ipynb` for training/testing workflows.

3️⃣ **Custom dataset creation:**
- Binary, ternary, and multilabel splits are included in the notebook.

---

## 📂 Repository Structure

```
├── data/                     # Dataset (not included)
├── ExDark_annotator.py       # Annotation processing script
├── csc245_proj3.ipynb        # Jupyter notebook (main workflow)
├── image_annotations.csv     # Raw annotations (output)
├── image_annotations_merged.csv # Merged annotations
├── README.md
├── venv.txt                  # Environment dependencies
└── TrainedModels/            # Pretrained models (optional)
```

---

## 👥 Contributors

- **Yuesong Huang** (yhu116@u.rochester.edu)
- **Wentao Jiang** (wjiang20@u.rochester.edu)

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---
