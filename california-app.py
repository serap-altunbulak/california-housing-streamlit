import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
import pickle


st.title('California House Price Prediction App')

# loading the boston house price dataset
cal = fetch_california_housing(as_frame = True)
x = pd.DataFrame(data=cal.data, columns=cal.feature_names)
y = pd.DataFrame(data=cal.target, columns=cal.target_names)
cal = pd.concat([x,y], axis=1)

st.subheader('Raw data')
st.dataframe(data=cal)

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Longitude = st.sidebar.slider('Longitude', x.Longitude.min(), x.Longitude.max(), x.Longitude.mean())
    Latitude = st.sidebar.slider('Latitude', x.Latitude.min(), x.Latitude.max(), x.Latitude.mean())
    MedInc = st.sidebar.slider('MedInc', x.MedInc.min(), x.MedInc.max(), x.MedInc.mean())
    HouseAge = st.sidebar.slider('HouseAge', x.HouseAge.min(), x.HouseAge.max(), x.HouseAge.mean())
    AveRooms = st.sidebar.slider('AveRooms', x.AveRooms.min(), x.AveRooms.max(), x.AveRooms.mean())
    AveBedrms = st.sidebar.slider('AveBedrms', x.AveBedrms.min(), x.AveBedrms.max(), x.AveBedrms.mean())
    Population = st.sidebar.slider('Population', x.Population.min(), x.Population.max(), x.Population.mean())
    AveOccup = st.sidebar.slider('AveOccup', x.AveOccup.min(), x.AveOccup.max(), x.AveOccup.mean())

    data = {'Longitude': Longitude,
            'Latitude': Latitude,
            'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup}

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


model = pickle.load(open('model.pkl','rb'))

# apply model to make prediction
prediction = model.predict(df)

st.header('Prediction of MedHouseVal')
st.write(prediction)
st.write('---')


st.header('SHAP Evaluation of The Model')
st.write("Here the features are ordered from the highest to the lowest effect on the prediction. It takes in account the absolute SHAP value, so it does not matter if the feature affects the prediction in a positive or negative way.")
st.image('images/bar.png', caption='BAR PLOT')
st.write("On the beeswarm the features are also ordered by their effect on prediction, but we can also see how higher and lower values of the feature will affect the result.\nAll the little dots on the plot represent a single observation. The horizontal axis represents the SHAP value, while the color of the point shows us if that observation has a higher or a lower value, when compared to other observations.\nIn this example, higher latitudes and longitudes have a negative impact on the prediction, while lower values have a positive impact.")
st.image('images/beeswarm.png', caption='BEESWARM')
st.write("Another way to see the information of the beeswarm is by using the violin plot.")
st.image('images/violin.png', caption='VIOLIN')