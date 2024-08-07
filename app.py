import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title='Suicide Rate Prediction App',
                   page_icon=':chart_with_upwards_trend:',
                   layout='wide',
                   initial_sidebar_state='expanded', )

PAGE_BG = 'https://www.wallpaperup.com/uploads/wallpapers/2017/01/17/1049490/9d5d3fb3b7c6c40cddc2f22a86b5b0c1.jpg'
st.markdown(f'<style>body{{background-image: url("{PAGE_BG}");background-size: cover;}}</style>',
            unsafe_allow_html=True)

#get options from dataset
df = pd.read_csv('data/final_version_data.csv')

country_list = list(df.country.unique())
gender_list = list(df.sex.unique())
age_group_list = list(sorted(df.age.unique()))
generation_list = list(df.generation.unique())


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


    # Apply the numerical scaler to the input data
    numerical_features = ['suicides_no', 'population', 'gdp_for_year', 'gdp_per_capita']
    input_data[numerical_features] = num_scaler.transform(input_data[numerical_features])

    return input_data


def make_prediction(input_data):
    # Make a prediction using the pre-trained XGBoost model
    prediction = model.predict(input_data)

    # Inverse transform the scaled target variable to get the original value
    prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))

    prediction = prediction[0][0]

    return prediction

# Define the input form

def input_form():
    st.sidebar.header('Input Parameters')

    with st.sidebar.form('input_form'):
        # Define the input fields
        country = st.selectbox('Country', country_list)
        sex = st.selectbox('Gender', gender_list)
        age_group = st.selectbox('Age Group', age_group_list)
        suicides_no = st.number_input('Number of Suicides', min_value=0)
        population = st.number_input('Population', min_value=0)
        hdi = st.number_input('Human Development Index', min_value=0.0, max_value=1.0, step=0.01)
        gdp_for_year = st.number_input('GDP for Year', min_value=0)
        gdp_per_capita = st.number_input('GDP per Capita', min_value=0)
        generation = st.selectbox('Generation', generation_list)

        # Define the submit button
        submitted = st.form_submit_button('Predict')

    if submitted:
        # Preprocess the input data
        input_data = preprocess_input(country, sex, age_group, suicides_no, population, hdi, gdp_for_year,
                                      gdp_per_capita, generation)

        # Make predictions using the preprocessed data
        prediction = make_prediction(input_data)

        return [country, sex, age_group, suicides_no, population, 
        hdi, gdp_for_year, gdp_per_capita, generation, prediction]
    
    return [" "]*10

# Define the main function
def main():

    st.title('Machine Learning Project')
    st.header('Suicide Rate Predictor')

    st.markdown("---")
    st.subheader("Created by:")
    st.markdown("""
    - **Souphaxay Naovalath** (3CS1)
    - **Thongchan Laothang** (3CS1)
    """)
    st.markdown("---")
    st.subheader('Input Parameters')

    lst = input_form()
    st.write('##### Country = ', lst[0])
    st.write('##### Sex = ', lst[1])
    st.write('##### Age Group = ', lst[2])
    st.write('##### Number of Suicides = ', str(lst[3]))
    st.write('##### Population = ', str(lst[4]))
    st.write('##### HDI = ', str(lst[5]))
    st.write('##### GDP for year = ', str(lst[6]))
    st.write('##### GDP per capita = ', str(lst[7]))
    st.write('##### Generation = ', lst[8])

    st.write('--------------')

    st.subheader('Prediction')
    if lst[0] == " ":
        st.warning(f'##### Predicted Suicide Rate: 0 per 100k people')
    else:
        prediction_formatted = '{:.2f}'.format(lst[9])
        st.warning(f'##### Predicted Suicide Rate: {prediction_formatted} per 100k people')

if __name__ == '__main__':
    main()

