# 🌤️ Weather Forecasting Pipeline with Apache Airflow
This project implements a complete machine learning pipeline for weather forecasting, orchestrated with **Apache Airflow**. It automates the end-to-end process from data extraction to prediction, including transformation and model training steps.

## Project Structure
- dags/: Airflow DAGs to automate data extraction, transformation, and model training.
- dataset/: Contains raw and processed weather data.

## How to run
### 1. Clone the repository
    git clone https://github.com/TerryNguyen1403/ML-Pipeline.git
    cd ML-Pipleline
### 2. Create a python virtual machine
    python -m venv env
    #Activate venv with Terminal
    env\Scripts\activate
### 3. Install dependencies
    pip install -r requirements.txt
### 4. Run extraction scripts in the extract/ directory to get a dataset from Open-meteo by calling its API.
### 5. Transform data
- Use the notebook script in transform/ directory to preprocess the data.
### 6. Train the model
- Rung the Training.py script in traning/ directory to feed the transformed dataset into GaussianNB, Random Forest Classifier, Support Vector Classifier (SVC) for training, evaluating and selecting a best model for predict new observations. The best model will be saved as a file name 'best_model.pkl' in predict/ directory.
### 7. Make predictions
- Run the script in predict/ directory to predict new observations.
