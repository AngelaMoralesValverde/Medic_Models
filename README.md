# 🧠 Brain Disease Detection with Deep Learning

> **A comparative Deep Learning study for automated brain disease classification from MRI images, with a focus on medical Computer Vision, model benchmarking, and explainable AI.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/)
[![Computer Vision](https://img.shields.io/badge/Domain-Medical%20Computer%20Vision-purple)](#)
[![Medical AI](https://img.shields.io/badge/AI-Medical%20Imaging-red)](#)
[![Status](https://img.shields.io/badge/Project-Research%20%26%20Development-green)](#)


# 📚 Base Datasets
- https://www.kaggle.com/datasets/alifatahi/multi-class-neurological-disorder-mcnd-dataset
- https://www.kaggle.com/datasets/shayalvaghasiya/ntua-prakinson
- https://www.kaggle.com/datasets/orvile/multiple-sclerosis-brain-mri-lesion-segmentation
- https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset
- https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans
- https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset
---

## 📌 Overview

**Medic_Models** is a medical Computer Vision project focused on the automatic classification of neurological conditions from **brain MRI images** using Deep Learning.

The project investigates and compares multiple state-of-the-art image classification architectures in order to understand how different CNN and Transformer-based approaches perform on medical imaging data.

The main objective is not simply to train a classifier, but to build a structured experimental pipeline covering:

* Medical image preprocessing
* Dataset preparation and transformation
* Transfer learning
* Deep Learning model training
* Multi-class classification
* Model evaluation
* Comparative benchmarking
* Error analysis
* Explainable AI
* Medical image visualization

The project is designed as an applied **AI for Healthcare** study, combining practical Machine Learning engineering with medical imaging concepts.

---

## 🎯 Project Goals

The project addresses the following question:

> **How effectively can modern Deep Learning architectures distinguish between different neurological conditions using brain MRI images?**

The main goals are:

1. Build a reproducible medical image classification pipeline.
2. Prepare heterogeneous brain MRI datasets for Deep Learning.
3. Compare different CNN and Transformer architectures.
4. Evaluate models using clinically relevant classification metrics.
5. Analyze model errors rather than relying exclusively on accuracy.
6. Investigate visual explainability techniques.
7. Explore the potential of AI-assisted medical image analysis.
8. Establish a foundation for a future multimodal medical AI system combining imaging with structured clinical data.

---

# 🧠 Problem Definition

Medical imaging contains complex visual patterns that can be difficult to identify consistently, particularly when several neurological conditions present overlapping characteristics.

This project frames brain disease detection as a **multi-class image classification problem**.

Given a brain MRI image:

```text
                    Brain MRI
                       │
                       ▼
              Image preprocessing
                       │
                       ▼
             Deep Learning model
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    Predicted class           Class probabilities
          │
          ▼
   Model evaluation
          │
          ▼
 Explainability / visual analysis
```

The model produces a predicted diagnostic category together with class probabilities, allowing the performance of different architectures to be compared.

> ⚠️ **Important:** This project is intended for research and educational purposes. It is **not a clinically validated diagnostic system** and should not be used to make medical decisions.

---

# 🏗️ Model Architectures

Five Deep Learning architectures are investigated in the current repository:

| Model                        | Architecture          | Category    | Main Motivation                                       |
| ---------------------------- | --------------------- | ----------- | ----------------------------------------------------- |
| **ResNet50**                 | Residual CNN          | CNN         | Strong baseline and residual learning                 |
| **DenseNet121**              | Densely Connected CNN | CNN         | Feature reuse and efficient gradient propagation      |
| **EfficientNetB0**           | Compound-scaled CNN   | CNN         | Accuracy/efficiency trade-off                         |
| **ConvNeXtLarge**            | Modernized CNN        | CNN         | High-capacity modern convolutional architecture       |
| **Vision Transformer (ViT)** | Transformer           | Transformer | Global attention and patch-based image representation |

### Why compare different architectures?

Medical imaging problems can behave very differently from conventional natural-image benchmarks.

Comparing several architectures makes it possible to investigate:

* Local vs. global feature extraction
* CNN vs. Transformer representations
* Model capacity vs. computational cost
* Feature reuse
* Transfer-learning behaviour
* Generalization across disease categories
* Classification errors between visually similar conditions

This makes the project more than a single-model implementation: it is a **comparative Deep Learning study**.

---

# 🔬 Medical Imaging Pipeline

The project includes preprocessing utilities for transforming medical imaging data into formats suitable for Deep Learning.

## NIfTI → 2D MRI slices

Medical imaging datasets can contain volumetric MRI data stored in **NIfTI (`.nii` / `.nii.gz`) format**.

The repository includes a preprocessing script that:

1. Loads NIfTI volumes.
2. Reads the volumetric data.
3. Normalizes intensity values.
4. Extracts individual axial slices.
5. Converts the slices into image files.
6. Organizes the resulting images by patient.

Conceptually:

```text
NIfTI volume
     │
     ▼
Intensity normalization
     │
     ▼
3D MRI volume
     │
     ├── Slice 1
     ├── Slice 2
     ├── Slice 3
     ├── ...
     └── Slice N
           │
           ▼
        2D images
           │
           ▼
     Deep Learning dataset
```

This enables conventional image-classification architectures to operate on MRI slices.

---

# 🗂️ Dataset Preparation

The project combines publicly available datasets covering different neurological disorders and brain abnormalities.

The repository currently references datasets including:

* **Multi-Class Neurological Disorder (MCND) Dataset**
* **Parkinson's-related MRI data**
* **Multiple Sclerosis brain MRI lesion data**
* **Brain Cancer MRI Dataset**
* **Brain Tumor MRI Scans**
* **Brain Tumors Dataset**

Because these datasets originate from different sources, an important part of the project is preparing and organizing them into a consistent classification pipeline.

### Dataset challenges

Medical datasets often present challenges such as:

* Different image formats
* Different acquisition protocols
* Different resolutions
* Different class distributions
* Different labeling conventions
* Patient-level dependencies
* Heterogeneous image preprocessing

These factors make dataset preparation an important part of the experiment rather than a simple preliminary step.

---

# 🔄 End-to-End Workflow

The complete experimental workflow can be summarized as:

```text
┌─────────────────────────┐
│ Public Medical Datasets │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Data Cleaning &         │
│ Dataset Organization    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ MRI Preprocessing       │
│ NIfTI → 2D slices       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Image Resizing &        │
│ Normalization           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Train / Validation /    │
│ Test Pipeline           │
└────────────┬────────────┘
             │
             ▼
      ┌──────┴──────┐
      │             │
      ▼             ▼
   CNN Models     ViT
      │             │
      └──────┬──────┘
             ▼
┌─────────────────────────┐
│ Performance Evaluation  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Error Analysis &        │
│ Explainability          │
└─────────────────────────┘
```

---

# 📊 Model Evaluation

Model performance should be evaluated using multiple complementary metrics rather than accuracy alone.

The project investigates standard classification metrics such as:

### Accuracy

Measures the overall percentage of correctly classified samples.

### Precision

Measures how many samples predicted as a given class actually belong to that class.

### Recall

Measures how many samples belonging to a class are successfully detected.

### F1-score

Provides a balance between precision and recall.

### Confusion Matrix

Provides a class-by-class view of the model's errors.

This is particularly important in medical classification because a high overall accuracy can hide poor performance on minority or visually similar classes.

A typical evaluation workflow is:

```text
Test Dataset
     │
     ▼
Model Predictions
     │
     ├── Accuracy
     ├── Precision
     ├── Recall
     ├── F1-score
     └── Confusion Matrix
```

---

# 🔎 Explainable AI

Medical AI should ideally provide more than a prediction.

One of the central research interests of this project is **model interpretability**: understanding which regions of an MRI image influence the model's prediction.

Explainability techniques can provide visual evidence of where a model is focusing when making a classification.

Examples include:

* **Grad-CAM / Grad-CAM++** for convolutional architectures
* **Layer-CAM** for CNN-based models
* **Attention Rollout** for Vision Transformers

Conceptually:

```text
MRI Image
    │
    ▼
Neural Network
    │
    ▼
Prediction
    │
    ▼
Activation / Attention analysis
    │
    ▼
Heatmap
    │
    ▼
Visual interpretation
```

This is particularly relevant to medical AI because interpretability can help researchers investigate whether a model is learning meaningful anatomical or pathological patterns rather than exploiting irrelevant visual correlations.

> Explainability maps should be considered **model-attribution visualizations**, not proof that a model has identified a clinically meaningful lesion.

---

# 📁 Repository Structure

```text
Medic_Models/
│
├── BrainDiseaseDetection.ipynb
│   └── Brain disease detection experiments
│
├── Medic_Models.ipynb
│   └── Model training and comparative experiments
│
├── prep_dataset.py
│   └── Dataset preparation utilities
│
├── paso_nii_a_jpg.py
│   └── NIfTI → JPG MRI conversion
│
├── README.md
│   └── Project documentation
│
├── Asistente de detección de enfermedades cerebrales
│   ├── Resumen.pdf
│   └── Full project documentation.pdf
│
└── ...
```

---

# 🧪 Experimental Design

The project follows a comparative experimental methodology.

Rather than assuming that one architecture is optimal, multiple models are trained and evaluated under comparable conditions.

The experimental process can therefore be viewed as:

```text
              Same Problem
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    ResNet50   DenseNet121  EfficientNet
       │           │           │
       └───────────┼───────────┘
                   │
             ConvNeXtLarge
                   │
                   ▼
             Vision Transformer
                   │
                   ▼
          Comparative Analysis
```

The final objective is to understand **which architectural characteristics are most suitable for this particular medical imaging problem**, rather than simply selecting a model based on a single metric.

---

# 💻 Technologies

The project uses the following technologies and concepts:

### Programming

* Python
* Jupyter Notebook

### Deep Learning

* TensorFlow / Keras
* Convolutional Neural Networks
* Transfer Learning
* Vision Transformers

### Computer Vision

* MRI preprocessing
* Image normalization
* Image resizing
* Medical image conversion
* Image classification

### Medical Imaging

* NIfTI
* MRI
* Neurological disease classification

### Machine Learning Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrices
* Classification reports

### Explainable AI

* Grad-CAM
* Grad-CAM++
* Layer-CAM
* Attention-based visualization

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/AngelaMoralesValverde/Medic_Models.git
cd Medic_Models
```

## 2. Create a Python environment

Using `venv`:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

## 3. Install dependencies

Install the required packages according to the environment used for the notebooks.

Typical dependencies include:

```bash
pip install tensorflow
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install nibabel
```

Additional packages may be required depending on the specific experiment.

## 4. Prepare the dataset

Download the relevant public datasets and organize them according to the directory structure expected by the notebooks.

For NIfTI-based datasets, the preprocessing utility can be used to convert volumetric data into 2D slices.

## 5. Run the notebooks

Open:

```text
Medic_Models.ipynb
```

or:

```text
BrainDiseaseDetection.ipynb
```

and execute the experiments sequentially.

---

# 📈 Results

The repository is structured around comparative evaluation of multiple Deep Learning architectures.

The most important result is therefore not simply:

> "Which model has the highest accuracy?"

but rather:

> **Which model provides the best balance between predictive performance, generalization, computational requirements, and interpretability for brain MRI classification?**

When reporting final experiments, the recommended comparison includes:

| Model          | Accuracy | Precision | Recall | F1 | Parameters | Inference Cost |
| -------------- | -------: | --------: | -----: | -: | ---------: | -------------: |
| ResNet50       |        — |         — |      — |  — |          — |              — |
| DenseNet121    |        — |         — |      — |  — |          — |              — |
| EfficientNetB0 |        — |         — |      — |  — |          — |              — |
| ConvNeXtLarge  |        — |         — |      — |  — |          — |              — |
| ViT            |        — |         — |      — |  — |          — |              — |

> Final benchmark values should be populated from the controlled test-set experiments rather than copied from training logs.

---

# 🧩 Key Engineering Challenges

This project involves several challenges that are particularly relevant to real-world Machine Learning engineering.

### 1. Heterogeneous medical data

Datasets obtained from different sources can have substantially different characteristics.

### 2. Medical image preprocessing

MRI data may originate as volumetric NIfTI files rather than conventional RGB images.

### 3. Class imbalance

Different diseases may have very different numbers of available samples.

### 4. Model comparison

Different architectures have different computational requirements and inductive biases.

### 5. Interpretability

A highly accurate model is not automatically a trustworthy medical model.

### 6. Generalization

Strong performance on a single dataset does not necessarily imply strong performance on images from another institution, scanner, acquisition protocol, or patient population.

These challenges make medical AI substantially different from a conventional image-classification project.

---

# 🔬 Research Perspective

This project can be viewed as a foundation for a broader **multimodal medical AI assistant**.

A future architecture could combine:

```text
                 Patient Data
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     MRI Images             Clinical Data
          │                       │
          ▼                       ▼
   Vision Model              ML Model
          │                       │
          └───────────┬───────────┘
                      ▼
              Multimodal Fusion
                      │
                      ▼
              Clinical Decision
                 Support
```

The long-term objective would be to combine:

* Medical imaging
* Structured clinical information
* Probabilistic predictions
* Model explainability
* Comparative model evaluation

into a unified decision-support framework.

---

# ⚠️ Limitations

Several limitations should be considered before interpreting the results clinically.

### Dataset limitations

Public datasets may not represent the full diversity of real-world clinical populations.

### Domain shift

MRI acquisition protocols, scanners, institutions and preprocessing pipelines can differ significantly.

### 2D representation

Converting volumetric MRI data into individual 2D slices can result in loss of three-dimensional contextual information.

### Dataset composition

When combining datasets from different sources, differences in acquisition and labeling can introduce hidden biases.

### Clinical validation

The models have not undergone prospective clinical validation and should not be considered medical devices or diagnostic systems.

### Explainability limitations

Heatmaps and attention visualizations provide insight into model behaviour, but they do not establish clinical causality.

---

# 🛠️ Future Improvements

The project provides a foundation for several future research directions.

## Multimodal Learning

Combine MRI images with structured patient information such as:

* Age
* Sex
* Clinical measurements
* Symptoms
* Laboratory results
* Other relevant metadata

## 3D Deep Learning

Instead of classifying independent 2D slices, future versions could use:

* 3D CNNs
* 3D Vision Transformers
* Volumetric segmentation
* Slice aggregation strategies

## Stronger Validation

Future experiments should include:

* Patient-level splitting
* Cross-validation
* External validation datasets
* Cross-dataset evaluation
* Calibration analysis

## Advanced Explainability

Potential extensions include:

* Integrated Gradients
* SHAP
* Occlusion analysis
* Attention visualization
* Counterfactual explanations

## Model Deployment

The trained models could eventually be exposed through:

* REST APIs
* Streamlit applications
* FastAPI services
* Docker containers
* Cloud inference endpoints

## Experiment Tracking

Future iterations could integrate:

* TensorBoard
* MLflow
* Weights & Biases
* Automated experiment tracking

---

# 📚 Data Sources

The project uses publicly available datasets from sources including Kaggle.

The datasets referenced by the project include:

* Multi-Class Neurological Disorder Dataset
* NTUA Parkinson's Dataset
* Multiple Sclerosis Brain MRI Lesion Segmentation Dataset
* Brain Cancer MRI Dataset
* Brain Tumor MRI Scans
* Brain Tumors Dataset

Dataset ownership, licensing and usage conditions remain with their respective providers and authors.

Users reproducing the experiments should download the datasets directly from their original sources and comply with their respective licenses.

---

# 📄 Project Documentation

The repository also contains the project's supporting documentation:

* **Project Summary**
* **Full Project Documentation**
* **Experimental notebooks**

These documents provide additional context regarding the motivation, methodology and development of the system.

---

# 🎓 Academic Context

This project was developed as an applied research project in **Artificial Intelligence, Machine Learning and Medical Computer Vision**.

It combines theoretical concepts with practical implementation, including:

* Deep Learning
* Transfer Learning
* Computer Vision
* Medical Image Processing
* Transformer architectures
* Model evaluation
* Explainable AI
* Experimental comparison

---

# 👩‍💻 Author

## Angela Morales Valverde

AI / Machine Learning enthusiast focused on:

* Artificial Intelligence
* Deep Learning
* Computer Vision
* Medical AI
* Data Science
* Machine Learning Engineering

This repository represents practical work in designing, training and evaluating Deep Learning systems for real-world problems.

---

# ⭐ Why This Project Matters

Medical AI is not only about achieving a high score on a benchmark.

A useful medical AI system must consider:

**Performance → Robustness → Interpretability → Generalization → Clinical relevance**

This project explores that complete pipeline by comparing multiple modern architectures and investigating how their predictions can be evaluated and interpreted.

The ultimate goal is to move from:

> **"Can a neural network classify this MRI?"**

towards:

> **"Can we build an AI system whose predictions are measurable, explainable, reproducible and potentially useful as part of a clinical decision-support workflow?"**

---

# 📌 Disclaimer

This repository is intended **for research and educational purposes only**.

The models developed in this project have not been clinically validated and must not be used as a substitute for professional medical diagnosis, clinical judgment, or treatment decisions.

No medical decision should be made solely on the basis of predictions generated by these models.

---

## ⭐ If you find this project interesting

Feel free to explore the notebooks, preprocessing pipeline and model experiments.

Feedback, discussion and research collaboration are welcome.
