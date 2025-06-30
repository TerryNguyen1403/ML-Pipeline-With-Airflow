import openmeteo_requests

import os
import pandas as pd
import requests_cache
from retry_requests import retry

# Import models
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Import metrics
from sklearn.metrics import accuracy_score

# Import joblib for saving and loading models
import joblib

import pendulum

from airflow.sdk import dag,task

@dag(
    schedule='@weekly',
    start_date=pendulum.datetime(2025,6,20),
    catchup=False,
    tags=['ML-Pipeline']
)
def ETL():
    @task
    def extract_weather_data():
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 10.823,
            "longitude": 106.6296,
            "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", "daylight_duration", "sunshine_duration",
                      "rain_sum", "showers_sum", "precipitation_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant"],
            "timezone": "Asia/Bangkok",
            "past_days": 7,
            "forecast_days": 1
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_weather_code = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
        daily_apparent_temperature_max = daily.Variables(3).ValuesAsNumpy()
        daily_apparent_temperature_min = daily.Variables(4).ValuesAsNumpy()
        daily_daylight_duration = daily.Variables(5).ValuesAsNumpy()
        daily_sunshine_duration = daily.Variables(6).ValuesAsNumpy()
        daily_rain_sum = daily.Variables(7).ValuesAsNumpy()
        daily_showers_sum = daily.Variables(8).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(9).ValuesAsNumpy()
        daily_precipitation_hours = daily.Variables(10).ValuesAsNumpy()
        daily_wind_speed_10m_max = daily.Variables(11).ValuesAsNumpy()
        daily_wind_gusts_10m_max = daily.Variables(12).ValuesAsNumpy()
        daily_wind_direction_10m_dominant = daily.Variables(13).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
        )}

        daily_data["weather_code"] = daily_weather_code
        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
        daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
        daily_data["daylight_duration"] = daily_daylight_duration
        daily_data["sunshine_duration"] = daily_sunshine_duration
        daily_data["rain_sum"] = daily_rain_sum
        daily_data["showers_sum"] = daily_showers_sum
        daily_data["precipitation_sum"] = daily_precipitation_sum
        daily_data["precipitation_hours"] = daily_precipitation_hours
        daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
        daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
        daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant

        daily_dataframe = pd.DataFrame(data = daily_data)

        # Create file path
        file_name = 'dataset/weather_data.csv'
        try:
            if os.path.exists(file_name):
                print(f"{file_name} is already exist! Appending new observations...")
                daily_dataframe.to_csv(file_name, mode='a', header=False, index=False, encoding='utf-8')
                print('Done...!')
            else:
                print(f"Creating dataframe at {file_name}...")
                daily_dataframe.to_csv(file_name, index=False, encoding='utf-8')
                print('Done...!')
        except Exception as e:
            print(f"Error: {e}")
    
    @task
    def transform_weather_data():
        """Classify weather based on WMO codes"""
        def classify_weather(code):
            if 0 <= code <= 3:
                return "Clear sky"
            elif 4 <= code <= 48:
                return "Fog"
            elif 49 <= code <= 55:
                return "Drizzle"
            elif 56 <= code <= 57:
                return "Freezing Drizzle"
            elif 58 <= code <= 65:
                return "Rain"
            elif 66 <= code <= 67:
                return "Freezing Rain"
            elif 68 <= code <= 75:
                return "Snow fall"
            elif 76 <= code <= 77:
                return "Snow grains"
            elif 78 <= code <= 82:
                return "Rain showers"
            elif 83 <= code <= 86:
                return "Snow showers"
            else:
                return "Thunderstorm"
            
        # Load dataframe
        df = pd.read_csv('dataset/weather_data.csv')

        # Adding label column based on weather code
        df["status"] = df["weather_code"].apply(classify_weather)

        # Dropping date and weather_code columns
        df = df.drop(columns=['date', 'weather_code'], axis=1)

        # Saving transformed data
        transformed_file_name = 'dataset/transformed_data.csv'
        try:
            if os.path.exists(transformed_file_name):
                print(f"{transformed_file_name} is already existed..! Appendind new transformed observations into a csv file")
                df.to_csv(transformed_file_name, mode='a', index=False,header=False,encoding='utf-8')
                print("Appended successfully!!!")
            else:
                print(f"Creating a transformed_data.csv")
                df.to_csv(transformed_file_name, index=False,encoding='utf-8')
                print("Created successfully!!!")
        except Exception as e:
            print(f"Error: {e}")
    
    @task
    def training_model():
        # Access dataframe
        df = pd.read_csv('dataset/transformed_data.csv')

        # Split into X and y
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Encoding categorical data
        # from sklearn.preprocessing import LabelEncoder
        # le = LabelEncoder()
        # y = le.fit_transform(y)

        #Split data into train set and test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define models
        models = {
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=10, criterion='gini', random_state=0),
            'Support Vector Classifier': SVC(random_state=0)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy

        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]

        best_model.fit(X_train, y_train)

        joblib.dump(best_model, 'model/best_model.pkl')
        # joblib.dump(scaler, 'predict/standard_scaler.pkl')

    extractTask = extract_weather_data()
    transformTask = transform_weather_data()
    trainTask = training_model()

    trainTask

ETL()