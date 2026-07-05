# AutoGluon Masterclass: Applied Machine Learning & Deep Learning

![AutoGluon Course Banner](BannerAutoGlounCourse.png)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-0.8+-orange.svg?style=flat-square)](https://auto.gluon.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

An end-to-end practical showcase of automated machine learning (AutoML) using **AutoGluon**—Amazon's state-of-the-art library for training highly accurate machine learning models with just a few lines of code. 

This repository contains a structured, modular set of Jupyter notebooks covering real-world machine learning paradigms: **Tabular Data, Multimodal Fusion (Text + Images + Tabular), Time-Series Forecasting, and Computer Vision**.

---

## 🚀 Key Modules & Project Architecture

The repository is organized into four core directories, each representing a distinct AutoML application domain:

### 1. 📊 Tabular Data (`00-Tabular-Data`)
Advanced tabular modeling using automated feature engineering, ensemble selection, and multi-layer stacking.
* **Key Tasks**: Binary classification, multi-class classification, regression, and hyperparameter tuning presets.
* **Datasets**: Uber rides fare regression and hotel reservation cancellations.
* **Notebooks**:
  * [01-Classification.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/00-Tabular-Data/01-Classification.ipynb) - AutoGluon Tabular classification fundamentals.
  * [02-Regression.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/00-Tabular-Data/02-Regression.ipynb) - Building optimized regression architectures.
  * [03-Advanced.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/00-Tabular-Data/03-Advanced.ipynb) - Model ensembling, custom feature engineering, and custom metrics.

### 🧩 2. Multimodal Fusion (`01-Multi-Modal`)
Fusing heterogeneous data sources (text, raw images, and tabular columns) into unified deep learning backends.
* **Key Tasks**: Natural Language Processing (NLP) classification, Multimodal fusion, and custom embedding extraction.
* **Datasets**: PetFinder adoption prediction, Kindle reviews, and text sentiment analysis.
* **Notebooks**:
  * [00-Multi-Modal-Natural Language for MultiClass.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/01-Multi-Modal/00-Multi-Modal-Natural%20Language%20for%20MultiClass.ipynb)
  * [01-Multi-Modal-Natural-Language-Binary-Classification.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/01-Multi-Modal/01-Multi-Modal-Natural-Language-Binary-Classification.ipynb)
  * [02-MultiModal-Image-Text-and-Tabular.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/01-Multi-Modal/02-MultiModal-Image-Text-and-Tabular.ipynb) - Unified training on a mixture of tabular properties, textual descriptions, and images.

### 📈 3. Time Series Forecasting (`02-Time-Series-Forecasting`)
Univariate and multivariate forecasting using deep learning models (DeepAR, Temporal Fusion Transformers, etc.).
* **Key Tasks**: Time-series forecasting incorporating past covariates and known future covariates.
* **Datasets**: Air quality metrics, alcohol sales trends, and restaurant visitor traffic.
* **Notebooks**:
  * [00-Single-Variate-Time-Series-Forecasting.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/02-Time-Series-Forecasting/00-Single-Variate-Time-Series-Forecasting.ipynb) - Univariate baseline forecasting.
  * [01-Known-Covariates-MultiVariate-Time-Series-Forecasting.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/02-Time-Series-Forecasting/01-Known-Covariates-MultiVariate-Time-Series-Forecasting.ipynb) - Integrating known future covariates (e.g. holidays, weather predictions).
  * [02-Past-Covariates-MultiVariate-Time-Series-Forecasting .ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/02-Time-Series-Forecasting/02-Past-Covariates-MultiVariate-Time-Series-Forecasting%20.ipynb) - Incorporating historical past variables to boost forecast accuracy.

### 🖼️ 4. Computer Vision & Embeddings (`03-Images`)
Deep Learning models applied to image classification and utilizing pre-trained foundation models.
* **Key Tasks**: Image classification and zero-shot similarity using OpenAI's CLIP model.
* **Notebooks**:
  * [00-Image-Classification.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/03-Images/00-Image-Classification.ipynb) - Fine-tuning vision models on custom datasets.
  * [01-OpenAI-CLIP-with-Autogluon.ipynb](file:///Users/carlosibarra/projects/autoGluon-Course/03-Images/01-OpenAI-CLIP-with-Autogluon.ipynb) - Leveraging OpenAI CLIP to extract embeddings and perform zero-shot classification.

---

## 🛠️ Installation & Setup

This project uses modern python package management via `uv` or `pip`.

### Prerequisites
* Python 3.9, 3.10, or 3.11
* CUDA-compatible GPU (Highly recommended for Multimodal and Computer Vision tasks)

### Step-by-Step Run Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/c-ibarra/autoGluon-Course.git
   cd autoGluon-Course
   ```

2. **Create and activate a virtual environment:**
   Using `uv` (Recommended):
   ```bash
   uv venv
   source .venv/bin/activate
   ```
   Or standard `venv`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

---

## 🧪 AutoML Models Utilized
AutoGluon automatically tests and ensembles a diverse range of architectures:
* **Tabular**: LightGBM, XGBoost, CatBoost, Random Forests, Extra Trees, and custom PyTorch Neural Networks.
* **Text/Multimodal**: Transformer backends (DeBERTa, ELECTRA) paired with tabular feature channels.
* **Vision**: ResNet, EfficientNet, ConvNeXt, and ViT (Vision Transformers).
* **Time Series**: AutoARIMA, Prophet, DeepAR, TFT (Temporal Fusion Transformer), and PatchTST.

---

## 💼 Portfolio Integration
*This project is featured as part of my Applied Machine Learning Portfolio. For more projects, case studies, and to get in touch, please visit my homepage:*

👉 **[c-ibarra Portfolio Homepage](https://c-ibarra.github.io)**

---

*Developed by **[c-ibarra](https://github.com/c-ibarra)**.*
