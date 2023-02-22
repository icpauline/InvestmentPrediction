# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:39:19 2023

@author: Pauline
"""

import streamlit as st
import pandas as pd
import numpy as np
import chart_studio.plotly as plotly
from plotly import graph_objs as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import CustomBusinessDay


st.title('Investment Prediction App')
dataset =('AMAZON',  'APPLE',	'CISCO', 'GOOGLE', 'INFOSYS')
option = st.selectbox('Select dataset for prediction',dataset)
DATA_URL =('C:\\Users\\Pauline\\Desktop\\Intern\\Investment Prediction\\'+option+'.csv')

year = st.slider('Year of prediction:',1,5)
period = year * 365

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data... done!')

def plot_fig():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open",line_color='deepskyblue'))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close",line_color='dimgray'))
	fig.layout.update(title_text='Time Series data with Rangeslider',xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	return fig

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
	
# plotting the figure of Actual Data
plot_fig()

st.write("# Forecast")

# Preparing the data for prediction
data_pred = data[['Close']]


# load the h5 model
model = load_model('investment_lstm_model.h5')

#Scaling the inputs
scaler=MinMaxScaler(feature_range=(0,1))
data_pred = scaler.fit_transform(np.array(data_pred).reshape(-1,1))

def predictinvest(data_pred,period,model):
    
    b = len(data_pred)-100
    x_input = data_pred[b:].reshape(1,-1)
    #x_input = period.reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    lst_output = []
    n_steps= 100
    i=0
    while(i<period):
    
        if(len(temp_input)>100):
            #print(temp_input)
            x_input = np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            #print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            #print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    return lst_output

# make predictions using the loaded model
#predictions = model.predict(data_pred)
predictions = predictinvest(data_pred,period,model)
predictions = scaler.inverse_transform(predictions)

def predictionDF(predictions,period):
    # create a custom business day frequency
    bday = CustomBusinessDay(weekmask='Mon Tue Wed Thu Fri', holidays=['2023-02-01', '2029-02-01'])
    data = pd.date_range('2023-01-31', periods=period, freq=bday)
    #data['Date'] = data['Date'].dt.date
    df_date = pd.DataFrame({'Date':data})
    df_date = df_date.reset_index()
    #Creating prediction dataframe
    df_pred = pd.DataFrame({'Predictions': predictions.reshape(-1)})
    df_pred = df_pred.reset_index()
    #Merging the dataframes
    df_prediction = pd.merge(df_date, df_pred, on='index')
    return df_prediction

PredictedDF = predictionDF(predictions,period)

if st.checkbox('Show Forecast data'):
    st.subheader('Forecast data')
    st.write(PredictedDF)

def plot_fig2():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Present", line_color='dimgray'))
    fig.add_trace(go.Scatter(x=PredictedDF['Date'], y=PredictedDF['Predictions'], name="Forecast", line_color='blue'))
    fig.layout.update(title_text='Forecast Time Series data with Rangeslider',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig
plot_fig2()

