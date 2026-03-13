# Food Image Recognition and Approximate Calorie Estimation Using Transfer Learning

## Project Overview
This project explores whether a pretrained convolutional neural network (CNN) can recognise food images accurately enough to support approximate calorie and macronutrient estimation at the dish level.

Instead of estimating portion size or weight from images, the problem is simplified into two stages:
1. **Food image classification**
2. **Nutritional lookup**

A ResNet50 model pretrained on ImageNet is adapted to the **Food-101 dataset** using transfer learning. Once a dish is predicted, the label is mapped to a small nutritional table containing typical per-serving values for calories, protein, fat, and carbohydrates.

The goal of the project is not precise dietary analysis but to demonstrate how food recognition models can support **fast, approximate diet tracking**.

## Project Files

### `Food_Image_Recognition.ipynb`
Jupyter notebook containing the full implementation:
- Dataset loading and preprocessing
- PyTorch data pipeline (`Dataset` and `DataLoader`)
- Transfer learning with ResNet50
- Training baseline and fine-tuned models
- Evaluation on the Food-101 test set
- Visualisation of predictions and misclassifications
- Example nutritional lookup for predicted dishes

The notebook was executed using **Google Colab with an NVIDIA T4 GPU**.

### `Food_Image_Recognition_report.pdf`
The written project report describing:
- Problem motivation and research question
- Related work in food image recognition
- Methodology and modelling approach
- Experimental setup and training configuration
- Results and discussion
- Limitations and future work

## Requirements

To run the notebook locally you need:

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- pillow

Install the required libraries with:

```bash
pip install torch torchvision numpy pandas matplotlib pillow
```
## Dataset

The project uses the **Food-101 dataset**:

Bossard, L., Guillaumin, M., & Van Gool, L. (2014).  
*Food-101 – Mining Discriminative Components with Random Forests.*

Dataset size:
- **101 food categories**
- **101,000 images**

The dataset can be downloaded from:
- https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

It can also be accessed through a public Kaggle mirror.

## Model Summary

Architecture: **ResNet50 (ImageNet pretrained)**

Two training strategies were evaluated:

### 1. Frozen Feature Extractor
- Backbone frozen
- Only classification head trained

### 2. Partial Fine-Tuning
- Final residual block (`layer4`) unfrozen
- Classification head trained

Performance on the Food-101 test set:

| Model | Test Accuracy |
|------|------|
| Frozen ResNet50 | 0.6214 |
| Fine-tuned ResNet50 | 0.7997 |

## Limitations

The system provides **approximate dish-level estimates only**.

It does not:
- estimate portion size
- recognise multiple foods on the same plate
- handle mixed meals

Classification errors also propagate directly into calorie estimates.

## Author
Anastasia Alyoshkina  
Applied Data Science graduate from Noroff University College

- [LinkedIn ](https://www.linkedin.com/in/anastasiia-alyoshkina-68ba5929a/)
- Email: anastasia.alshkn@gmail.com
