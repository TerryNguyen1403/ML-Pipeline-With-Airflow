# 🌤️ Weather Forecasting Pipeline with Apache Airflow
This project implements a complete machine learning pipeline for weather forecasting, orchestrated with **Apache Airflow**. It automates the end-to-end process from data extraction to prediction, including transformation and model training steps.

## Prerequisites
- Docker and Docker Compose

## How to run
### 1. Clone the repository
    git clone https://github.com/TerryNguyen1403/ML-Pipeline-With-Airflow.git
    cd ML-Pipeline-With-Airflow
### 2. Build and Start Airflow with Docker Compose
    docker-compose up -d
This command will build the necessary Airflow images and start the services in detached mode.
### 3. Access the Airflow Web UI
- Once the containers are running, navigate to: http://localhost:8080
- The username and password is **airflow** by default
    
### 4. Enable and Run the DAG.
- In the DAGs list, find the DAG named ETL.

- Toggle the switch to Enable the DAG.

- Click the Play/Trigger button to run the pipeline manually.

You can monitor each task in the DAG:

- Click on the DAG name -> View the Graph View or Tree View -> Click on each task to see detailed logs, status, and outputs
