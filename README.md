[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10563750&assignment_repo_type=AssignmentRepo)
Project Instructions
==============================

This repo contains the instructions for a machine learning project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │   └── Visualization.ipynb        <- Contains visualizations from dataset
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── README.md      <- Youtube Video Link
    │   └── Final Report.pdf <- final report .pdf format 
    │   └── CP322 Project Presentation.pptx   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── dataset.py     <- Class containing functionality for handling datasets, preprocessing and overfitting regulation
       │
       ├── models         <- Class containing functionality for ML models including K-Nearest-Neighbors and Logistic Regression
       └── heart.csv      <- Dataset     

### Abstract - Heart Health detection
Heart failure is a severe and life-threatening illness that is responsible for the majority of deaths worldwide. According to WHO, approximately 17.9 million people die from heart disease each year, with patients often unaware of their health status, making them vulnerable to various cardiovascular diseases. Heart disease patients have a high mortality rate, with 55% of them dying within the first three years of diagnosis. Treatment is also quite expensive, accounting for about 4% of the annual healthcare system cost. An  machine learning algorithm would help address these challenges by automating the diagnosis process resulting in a reduction in costs, more accurate diagnoses and an improvement in treatment outcomes. The following project will handle this problem in the form of supervised binary classification and will include the classes yes and no (in regards to the diagnosis of heart disease). Machine learning models including Logistic Regression and K-Nearest-Neighbors will be utilized for the purpose of finding the best possible classifier for heart disease.