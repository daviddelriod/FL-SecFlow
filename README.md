# FL-SecFlow

This repository contains the code and resources for my research project called ***A privacy preserving federated learning flow with gradient leakage prevention*** in this case, applied for pneumonia detection. The project explores techniques to enable privacy-preserving and secure collaborative training of machine learning models across multiple nodes, providing notebooks and methods for the implementations.

<video src="https://github.com/daviddelriod/FL-SecFlow/assets/64251001/0f9c2b0e-7752-4978-bec1-e03636b27e10" controls="controls" style="max-width: 730px;">
</video>

## Abstract
Deep learning-based approaches for pneumonia detection from chest X-rays
have shown promising results. However, collecting and sharing medical data raises
concerns about patient privacy. Federated learning can help address these
concerns by allowing models to be trained on decentralized data without transmitting
sensitive patient information. However, federated learning is also vulnerable to
gradient leakage, which can reveal sensitive information about the local data
sources. Gradient leakage occurs when an attacker can infer delicate information
from the original data during federated learning by exploiting the gradients sent by
the devices during the parameter sharing. To address this issue, a privacy-preserving
federated learning framework for pneumonia detection from chest X-rays that
includes gradient leakage prevention measures is proposed. The presented approach
uses a combination of secure aggregation and encoding techniques to ensure that
gradients are not leaked during the federated training process. It is evaluated on a
publicly available chest X-ray dataset and demonstrate that this framework provides
competitive performance while protecting patient privacy. This work contributes to
the development of secure and privacy-preserving deep learning techniques for
medical image analysis and has important implications for preserving the accuracy
and accessibility of medical diagnosis.

Link: https://drive.google.com/file/d/1pO8PpGSX-Jh7oVKtdadTxhrQuHOc6wpb/view?usp=sharing

## Folders' structure

### 1. `director`
The `director` folder contains the implementation of the federated learning orchestrator. This component is responsible for coordinating the training process across different clients, aggregating model updates, and ensuring data privacy.

### 2. `envoy`
The `envoy` folder includes the implementation of the communication module for federated learning. The envoy facilitates secure communication between the central server and the clients, ensuring encrypted data transmission.

### 3. `gradient-leakage`
The `gradient-leakage` folder houses the code for implementing the defense mechanism against gradient leakage. This section explores the use of variational autoencoders (VAEs) to mitigate privacy risks during the federated learning process.

### 4. `utils`
The `utils` folder provides utility functions and helper scripts used across different parts of the project. It contains reusable code for data preprocessing, model evaluation, and other common tasks.

### 5. `workspace`
The `workspace` folder serves as the workspace for the project. It contains configuration files, data samples, and Jupyter notebooks for running experiments, visualizing results, and analyzing the performance of the proposed approaches.
