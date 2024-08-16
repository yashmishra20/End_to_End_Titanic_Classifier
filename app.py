import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pickled model and preprocessor
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Create the Streamlit app
st.title('Titanic Survival Prediction')

# Create input fields for user data
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, value=50.0)
embarked = st.selectbox('Embarkation Point', ['C', 'Q', 'S'])

# Create a dataframe from user input
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Add a prediction button
if st.button('Predict Survival'):
    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)

    # Display result
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.write('The passenger would likely survive.')
    else:
        st.write('The passenger would likely not survive.')
    
    st.subheader('Survival Probability:')
    st.write(f'{probability[0][1]:.2%}')

# Add some information about the project
st.sidebar.header('About')
st.sidebar.info('This app uses a Random Forest Classifier trained on the Titanic dataset to predict passenger survival.')