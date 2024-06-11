# AquaScan

## Overview of the Project

AquaScan is a cutting-edge application designed to predict water quality, ensuring access to clean and safe water for all. Utilizing a machine learning model built with TensorFlow, AquaScan leverages a FastAPI backend and a React frontend to deliver real-time predictions on water potability based on various water quality parameters.

## Problem Statement

Water quality is a critical issue as contaminated water can lead to severe health problems and diseases. By monitoring and predicting water quality, AquaScan helps prevent the consumption and use of dirty water, thereby safeguarding public health and reducing medical expenses associated with waterborne illnesses.

## The Dataset

The dataset used for this project is `water_potability.csv`, which contains 3,276 rows and 10 columns. The key features of the dataset include:

- PH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity
- Potability

AquaScan is trained to predict the potability of water based on the first nine features.

## The Solution

AquaScan processes the input water quality parameters and predicts whether the water is potable or not. This prediction is made using a Multi-Layer Perceptron Neural Network (MLP) model, which has been trained on historical water quality data.

## Architecture of the Model

The architecture of the machine learning model is a Multi-Layer Perceptron (MLP) Neural Network, consisting of:

- One input layer
- Two hidden layers
- One output layer

### Choice of Activations, Regularizers, and Optimizers

- **Activation Functions:** ReLU for the hidden layers and Sigmoid for the output layer.
- **Regularization:** L2 regularization, early stopping, confusion matrix, and dropouts were employed to enhance model performance and prevent overfitting.
- **Optimizer:** Adam optimizer.

## Performance Analysis

The performance of the model is evaluated based on its loss and accuracy metrics:

- **Loss Function:** Binary Cross-Entropy
- **Metrics:** Accuracy

Performance statistics:
- Accuracy: 0.6908
- Loss: 0.5948
- Validation Accuracy: 0.6585
- Validation Loss: 0.6135

## Libraries and Frameworks

AquaScan utilizes the following libraries and frameworks:

- Pandas (pd)
- NumPy (np)
- TensorFlow
- Matplotlib
- Scikit-learn (sklearn)

## FastAPI

The FastAPI backend serves as the core of AquaScan's API, facilitating real-time predictions. For interactive API documentation, refer to the Swagger UI (link to be added).

## React Frontend

The React frontend collects nine inputs from the user corresponding to the water quality features (excluding potability). These inputs are then sent to the backend for prediction.

## Get Started

To set up the project locally:

1. Clone the Git repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the respective directories for backend and frontend:
   ```
   cd AquaScan/backend
   cd AquaScan/frontend
   ```
3. Set up a virtual environment to manage dependencies:
   ```
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
4. Install the required modules:
   ```
   pip install -r requirements.txt
   ```

## How to Use the API for Prediction

The API is available on this FastAPI [Swagger](https://aquascan.onrender.com/docs#/), which was hosted through Render. To make predictions, send a POST request with the following nine float features:

- PH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

To this endpoint:

```
https://aquascan.onrender.com/predict
```

The endpoint will return a prediction of the potability.

## Authors/Contributors

AquaScan was developed by:
- [kayc0des](https://github.com/kayc0des)
- [DavidkingMazimpaka](https://github.com/DavidkingMazimpaka)
- [thedavidemmanuel](https://github.com/thedavidemmanuel)
- [ZigaLarissa](https://github.com/ZigaLarissa)

Students at African Leadership University.
