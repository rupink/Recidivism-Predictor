
# Fairness and Bias in Recidivism Prediction

This repository contains code for implementing a fairness and bias analysis on a recidivism prediction model. The code is written in Python and uses popular libraries like PyTorch and pandas.

## Overview

The purpose of this code is to explore fairness and bias issues in a recidivism prediction model. It uses a logistic regression model and an adversarial fairness model to predict whether a defendant will reoffend within two years based on various features, including age, gender, and charge descriptions. Additionally, it assesses the fairness of the model's predictions with respect to race and adjusts the threshold to achieve fairness.

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/recidivism-fairness.git
2. Navigate to the project directory:
    ```bash
    cd recidivism-fairness

3. Run the code using Python:
    ```bash
    python Main.ipynb
Examine the results and fairness metrics in the console output.


Code Explanation
The code is organized into several modules:

Train_Test_Loops.py: This script contains the main training and testing loops for the logistic regression and adversarial models.

Imports.py: Imports necessary libraries for the analysis.

LogisticRegression.py: Defines and trains a logistic regression model.

Data_Format.py: Formats the data, including one-hot encoding charge descriptions.

TrainandTestDataset.py: Defines custom datasets for training and testing.

Main_Adver_Network.py: Defines the adversarial fairness model.

Train_Advs_Main.py: Contains functions for training the adversarial and main networks.

Data Source
The dataset used in this analysis is available at the following URL:
Dataset on GitHub

Model Training
The logistic regression model is trained to predict recidivism based on features such as age, gender, and charge descriptions.

An adversarial fairness model is introduced to mitigate bias in predictions, achieving fairness across different races.

Results and Fairness Metrics
The code provides detailed metrics for fairness, including False Positive Rate (FPR) parity, for different racial groups. Below are the metrics for the tow groups.

African-American Group
True Positives: [Number]
True Negatives: [Number]
False Positives: [Number]
False Negatives: [Number]
Positive Predictive Value (PPV): [Value]%
Negative Predictive Value (NPV): [Value]%
False Positive Parity: [Value]%
Caucasian Group
True Positives: [Number]
True Negatives: [Number]
False Positives: [Number]
False Negatives: [Number]
Positive Predictive Value (PPV): [Value]%
Negative Predictive Value (NPV): [Value]%
False Positive Parity: [Value]%
These metrics demonstrate the fairness and bias mitigation achieved by the adversarial fairness model.

Conclusion
This code showcases an analysis of fairness and bias in a recidivism prediction model, providing insights into model performance and fairness across different racial groups.

Credits
This code was created by Rupin Khadwal.

License
This project is licensed under the MIT License - see the LICENSE file for details.
