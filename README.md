# Image Classification with Deep Learning: CIFAR-10

A comprehensive deep learning project that classifies images from the CIFAR-10 dataset using both traditional ML and a Convolutional Neural Network (CNN), demonstrating why deep learning dominates computer vision.

## Setup

1. Install Python 3.10+ (tested with Python 3.13)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open notebooks in order: `notebooks/01_Introduction.ipynb` through `05_Comparison_and_Results.ipynb`

## Notebook Order

| #   | Notebook                          | Description                                                        |
| --- | --------------------------------- | ------------------------------------------------------------------ |
| 1   | `01_Introduction.ipynb`           | Deep learning concepts: neural networks, CNNs, convolution         |
| 2   | `02_Data_Exploration.ipynb`       | CIFAR-10 EDA: visualizations, class distribution, pixel statistics |
| 3   | `03_Baseline_ML_Model.ipynb`      | Traditional ML baselines: Logistic Regression & SVM                |
| 4   | `04_CNN_Model.ipynb`              | Build, train, and evaluate a CNN                                   |
| 5   | `05_Comparison_and_Results.ipynb` | Side-by-side ML vs DL comparison and conclusions                   |

## Results

| Model               | Test Accuracy |
| ------------------- | ------------- |
| Logistic Regression | ~40%          |
| SVM (RBF Kernel)    | ~50%          |
| CNN (SimpleCNN)     | ~85-90%       |

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, matplotlib, seaborn, Jupyter
