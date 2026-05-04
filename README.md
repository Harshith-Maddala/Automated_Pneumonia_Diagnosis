# Deep Learning Based Detection of Thorax Diseases from Chest X-rays with Confidence Estimation

## Background
Thoracic diseases alone account for millions of deaths globally each year. Pneumonia alone remains as one of the leading causes of deaths, especially among older adults and immune compromised individuals. In the United States, chronic lower respiratory diseases collectively rank among the top ten leading causes of deaths. Globally, lower respiratory infections are responsible for over 2 million deaths per year, underscoring their persistent severity. The severity of thoracic diseases is not only in their mortality rates but also in their potential for rapid deterioration. Many of these conditions can progress quickly, requiring timely diagnosis and intervention to prevent complications such as sepsis, acute respiratory distress syndrome (ARDS), or multi-organ failure. In addition, broader burden includes recurrent hospitalizations, reduced quality of life, long-term disability, and increased strain on emergency and intensive care services. 
Given their frequency, severity, and potential for acute deterioration, thoracic diseases remain a critical public health challenge. Early and accurate detection is essential to improving patient outcomes, reducing preventable deaths, and strengthening healthcare delivery systems.

## Problem Statement
Every year in US, around 55,000 people die due to  pneumonia alone. Quick and accurate detection is crucial because delays or mistakes can lead to worsening health conditions or even death. Chest X-rays are often best tools to detect common lung, chest and other thoracic diseases. Recent Deep Learning Models can identify these diseases with high accuracy on large datasets, but their predictions can be overconfident and sometimes unreliable. Most current models do not clearly indicate how certain they are about a detection, which is a problem because understanding of how trustworthy the result is as important as detection of the diseases. This gap between high accuracy in tests and real-world reliability makes it important to build systems that are both accurate and provide clear, trustworthy confidence in their predictions.

## Scope of this project
This project focuses on detecting commonly known thorax diseases such as pneumonia, cardiomegaly, pleural effusion, pneumothorax, lung opacity, atelectasis, and consolidation from chest X-ray images using deep learning models. The diseases we can predict are based on the dataset we choose and cannot detect diseases which are not available in dataset. The study evaluates how accurately these diseases can be identified and, more importantly, how reliable the model’s confidence is when making these predictions. The scope further includes confidence estimation and uncertainty quantification to assess not only what the model predicts, but how certain it is about those predictions.

However, it is critical to emphasize that such models should not make final medical decisions in thoracic disease diagnosis or any other diagnosis. Medical imaging interpretation involves clinical history, laboratory findings, physical examination, and most importantly, a experienced physician, and these are the factors that AI models cannot provide. 

## Literature Review
1. Automated detection of pneumonia cases using deep transfer learning with paediatric chest X-ray images
Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC8506182/
This study aimed to detect pneumonia in children using chest X-rays by applying deep learning models with transfer learning. Four pre-trained convolutional neural networks: VGG19, DenseNet121, Xception, and ResNet50, were tested and DenseNet121 has performed better compared to others. Transfer learning helped overcome the challenge of limited data. 
Insights:
The study showed that models such as ResNet50 and DenseNet121 worked really well and could be used as base models for our project

2. Uncertainty quantification in multi-class image classification using chest X-ray images of COVID-19 and pneumonia
Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC11445153/
This paper tries to find the role of uncertainty quantification in multi-class chest X-ray classification, focusing on COVID-19, pneumonia, and normal cases. The authors compare Bayesian neural networks with practical deep learning–based uncertainty estimation methods, including Monte Carlo dropout and ensemble techniques. Their results show that ensemble-based approaches provide better calibrated predictions, improved robustness, and more reliable confidence estimates than standard Bayesian models.
Insights:
They explained in-detail on different ways to deal with uncertainty in data and also motivated us to use Monte Carlo dropout as one of the methods to determine confidence estimation.

3. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison
Link: https://stanfordmlgroup.github.io/competitions/chexpert/
This paper proposes CheXNet, a deep convolutional neural network based on DenseNet-121, for detecting pneumonia and other thoracic diseases from chest X-ray images, achieving performance comparable to radiologists on a large public dataset. It demonstrates that deep learning can effectively perform chest X-ray disease classification using image-level labels.
Insights:
They are one of the firsts to work on a large datasets on this topic and has achieved commendable results which are even better than experienced radiologists. 

4. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning
Link: https://arxiv.org/pdf/1711.05225
This paper introduced CheXNet, a 121-layer Dense Convolutional Network (DenseNet121) designed to detect pneumonia from chest X-ray images. The study aimed to develop a deep learning model capable of performing at or above radiologist level in pneumonia detection. CheXNet achieved performance comparable to, and slightly exceeding, practicing radiologists in terms of F1 score for pneumonia detection. 
Insights:
They also included the importance of modeling uncertainty in medical labels rather than ignoring or removing ambiguous cases, since different uncertainty handling strategies can significantly impact performance across pathologies.


Link to Best Models .pth file:
- https://drive.google.com/drive/folders/1ijOhKqQDK9fNlnbICkHVj-YRpkARBBHd?usp=drive_link

