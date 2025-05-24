# Fingerprint Matching System

This repository contains a collection of deep learning models for fingerprint matching, along with a simple web application for deployment and demonstration.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [License](#license)

## Overview
The project focuses on developing and evaluating various deep learning architectures for accurate fingerprint identification and verification. It includes implementations of several attention mechanisms and established CNN backbones.

## Features
- Multiple fingerprint matching models:
    - VGG-based
    - SENet (Squeeze-and-Excitation Networks)
    - CBAM (Convolutional Block Attention Module)
    - Self-Attention
    - Dual-Attention
- Pre-trained models:
    - MobileNet + Self-Attention
    - MobileNet + SENet
- A web application (`app.py`) for demonstrating model predictions.
- Supporting documentation (Report and Slides).

## Project Structure
```
.
├── LICENSE
├── README.md
├── deploy/
│   ├── app.py                           # Flask application for deployment
│   ├── mobilenet_plus+Self-Attention.h5 # Trained MobileNet + Self-Attention model
│   ├── mobilenet_plus+Snet.h5           # Trained MobileNet + SENet model
│   └── requirements.txt                 # Python dependencies for deployment
├── docs/
│   ├── Report.pdf                       # Project report
│   └── Slide.pdf                        # Project presentation slides
└── models/
    ├── fingerprint-cbam.py              # Fingerprint model with CBAM
    ├── fingerprint-dual.py              # Fingerprint model with Dual-Attention
    ├── fingerprint-self.py              # Fingerprint model with Self-Attention
    ├── fingerprint-senet.py             # Fingerprint model with SENet
    └── vgg-fingerprint.py               # VGG-based fingerprint model
```

## Models Implemented
The `models/` directory contains the Python scripts for the following architectures:
- **`vgg-fingerprint.py`**: A fingerprint recognition model based on the VGG architecture.
- **`fingerprint-senet.py`**: Implements Squeeze-and-Excitation Networks to improve channel interdependencies.
- **`fingerprint-cbam.py`**: Integrates the Convolutional Block Attention Module, which applies both spatial and channel-wise attention.
- **`fingerprint-self.py`**: Utilizes Self-Attention mechanisms to capture global dependencies within the fingerprint images.
- **`fingerprint-dual.py`**: Implements a Dual-Attention mechanism.

The `deploy/` directory contains pre-trained model weights:
- **`mobilenet_plus+Self-Attention.h5`**: A MobileNet-based architecture enhanced with a Self-Attention layer.
- **`mobilenet_plus+Snet.h5`**: A MobileNet-based architecture enhanced with an SENet layer. (Note: "Snet" likely refers to SENet).

## Setup and Installation
To set up the project and run the deployment application, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder-name>
    ```
    (Replace `<repository-url>` and `<repository-folder-name>` accordingly)

2.  **Navigate to the deployment directory:**
    ```bash
    cd deploy
    ```

3.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # Activate the virtual environment
    # On Windows (PowerShell):
    # .\venv\Scripts\Activate.ps1
    # On macOS/Linux:
    # source venv/bin/activate

    pip install -r requirements.txt
    ```

## Running the Application
The `app.py` in the `deploy` directory is a web application (likely Flask or Streamlit) to test the trained models.

1.  **Ensure you are in the `deploy` directory** and have installed the requirements.
2.  **Run the application:**
    ```bash
    python app.py
    ```
3.  Open your web browser and navigate to the URL provided by the application (e.g., `http://127.0.0.1:5000` or `http://localhost:8501`).

## License
This project is licensed under the terms of the `LICENSE` file. Please see the `LICENSE` file for more details.