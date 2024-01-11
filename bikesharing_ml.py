import streamlit as st
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from scipy.special import inv_boxcox
import pickle

# Load trained algorithm
def load_model():
    loaded_model = pickle.load(open('XGB-1-regression.pkl', 'rb'))

    return loaded_model

# Function to convert hr to time of day (morning, night, afternoon, or evening)
def time_of_day(hr):
    timeofday = []
    if hr >= 5 & hr <=12:
        timeofday = 1
    elif hr >= 13 & hr <= 17:
        timeofday = 2
    elif hr >= 18 & hr <= 21:
        timeofday = 3
    else :
        timeofday = 4
    
    return timeofday

# Function to change day name to number
def day_number(day):
    day_num = []
    if day == 'Monday':
        day_num = 0
    elif day == 'Tuesday':
        day_num = 1
    elif day == 'Wednesday':
        day_num = 2
    elif day == 'Thursday':
        day_num = 3
    elif day == 'Friday':
        day_num = 4
    elif day == 'Saturday':
        day_num = 5
    elif day == 'Sunday':
        day_num = 6

    return day_num

def season_code(season):
    x = []
    if season == 'Winter':
        x = 1
    elif season == 'Spring':
        x = 2
    elif season == 'Summer':
        x = 3
    elif season == 'Fall':
        x = 4
    
    return x

def main():

    st.title('Bikesharing Number of Bikes Prediction')

    hum = st.number_input('Insert normalize humidity value', min_value=0, max_value=1, step = 0.01)
    weathersit = st.selectbox('Weather Forecast', [1,2,3,4])
    holiday = st.selectbox('Holiday or not', [0, 1])
    season = st.selectbox('Select Season', ['Winter', 'Summer', 'Spring', 'Fall'])
    temp = st.number_input('Insert normalize temperature value', min_value=0, max_value=1, step = 0.01)
    hr = st.selectbox('Hour of the day', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    day = st.selectbox('Select day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday'])

    seasons = season_code(season)
    timeofdays = time_of_day(hr)
    days = day_number(day)
    
    predict_var = pd.DataFrame(data = {'hum':hum,
                                       'weathersit':weathersit,
                                       'holiday':holiday,
                                       'season':seasons,
                                       'temp':temp,
                                       'hr':hr,
                                       'day':days,
                                       'timeofday':timeofdays}
                                , index=[0])

    button = st.button('Predict')
    
    if button:
        result = inv_boxcox(load_model().predict(predict_var), lmbda = 0.33355718858939065)
        st.write(result)


if __name__ == '__main__':
    main()