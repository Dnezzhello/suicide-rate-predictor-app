import joblib
import pandas as pd

# load pre-trained models
model = joblib.load('models/XGBoost_model.joblib')
num_scaler = joblib.load('models/num_features_scaler.joblib')
target_scaler = joblib.load('models/target_scaler.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

def preprocess_input(country, sex, age_group, suicides_no,
                     population, hdi, gdp_for_year, gdp_per_capita,
                     generation):
    # Create a dictionary with the user input data
    data = {
        'country': country,
        'sex': sex,
        'age': age_group,
        'suicides_no': suicides_no,
        'population': population,
        'gdp_for_year': gdp_for_year,
        'gdp_per_capita': gdp_per_capita,
        'generation': generation,
        'hdi': hdi
    }

    # Convert the user input data to a DataFrame
    input_data = pd.DataFrame(data, index=[0])
    input_data.values.reshape(1, -1)

    # Encode categorical variables using the label encoder
    for cat, encoder in label_encoder.items():
        input_data[cat] = encoder.transform(input_data[cat])
    # input_data['country'] = label_encoder.transform(input_data['country'])
    # input_data['sex'] = label_encoder.transform(input_data['sex'])
    # input_data['age_group'] = label_encoder.transform(input_data['age_group'])
    # input_data['generation'] = label_encoder.transform(input_data['generation'])

    # Apply the numerical scaler to the input data
    numerical_features = ['suicides_no', 'population', 'gdp_for_year', 'gdp_per_capita']
    input_data[numerical_features] = num_scaler.transform(input_data[numerical_features])

    return input_data


def make_prediction(input_data):
    # Make a prediction using the pre-trained XGBoost model
    prediction = model.predict(input_data)

    # Inverse transform the scaled target variable to get the original value
    prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))

    return prediction