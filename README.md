# GitHub repository for project 3 in FYS-STK4155: Applied Data Analysis and Machine Learning
Group consisting of: 
  * Erik Berthelsen
  * Morten Taraldsten Brunes


## Description of dataset and project 3
In this project we consider both analysis of geophysical well-log data and classification of insects and pests in images. 


Well-log measurements are analysed with feed forward neural network to predict the sonic log (AC), which is an important indicator of lithology and porosity variations in the subsurface.

AgroPest-12 dataset downloaded from Kaggle are used for classification of insects and pests with convolutional neural networks.

## Installation and running the code
To install required packages:

Using pip

``
pip install -r requirements.txt
``


Using conda


``
conda install --yes --file requirements.txt
``


Once the appropriate environment has been created, and the necessary packages are up to date and code for processing well-data and AgroPest-12.

Code for processing well-log is run from jupyter notebook ffnn_well_log.ipynb.

Code for processing AgroPest-12 is run from jupyter notebook cnn_agropest12.ipynb.

Link provided in report are to project3 folder.

## Directory structure

├── project3

│   ├── cnn_results

│   │   ├── figures

│   │   │   ├── custom_150_epochs

│   │   │   ├── optuna_hyperparameter_search

│   │   │   ├── pretrained

│   ├── cnn_models

│   │   │   ├── custom_150_epochs

│   │   │   ├── optuna_hyperparameter_search

│   │   │   ├── pretrained

│   ├── code

│   │   ├── source

│   │   │   ├── cnn_model.py

│   │   │   ├── cnn_plotting.py

│   │   │   ├── cnn_quality_control.py

│   │   │   ├── cnn_retreive_images.py

│   │   │   ├── ffnn_model.py

│   │   │   ├── cnn_optuna.py

│   │   │   ├── well_data_preprocessing.py

│   │   ├── cnn_agropest12.ipynb

│   │   ├── ffnn_well_log.ipynb

│   ├── datasets

│   │   ├── agropest12

│   │   ├── well_data

│   ├── llm_documentation

│   ├── report


Code for processing well-log and AgroPest-12 are within source folder, where prefix ffnn* or cnn* determines which jupyter notebook the file is used in. Results and figures from different approaches in convolutional neural network (CNN)processing are in folder cnn_results. There are three different approaches: Custom architecture for CNN model, model based on Optuna hyperparameter search and using a pretrained model. Both datasets are available in datasets. Questions to large language model are in ll_documentation, while the report is available in report.