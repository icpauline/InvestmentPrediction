# InvestmentPrediction
## 1. Introduction:
### 1.1 Objective:
Data is power in today's world. Investment prediction projects using stock market data involve
analyzing historical stock market data to predict future trends and make informed investment
decisions. The goal is to use data science and machine learning techniques to build models that can
accurately predict future stock prices, identify profitable investment opportunities, and minimize
risks. The project will typically involve gathering and cleaning historical stock market data,
performing exploratory data analysis, selecting appropriate features, building and training machine
learning models, and evaluating their performance using various metrics.
### 1.2 Results:
The proposed model forecasts stock market data of the desired organization up to 5 years.
## 2. Project Scope:
The HLD documentation outlines the system's architecture, including the technology architecture,
application architecture (layers), application flow, and database architecture. The HLD employs
simple to moderately complex terms that system administrators should be able to understand.
## 3. Project methodology:
### 3.1 Architectural diagram
![image](https://user-images.githubusercontent.com/115965062/220602563-23f4630d-1fa7-4fe7-b65
### 3.2 System requirements
  o Windows 7 and above
  o SQL
  o PyCharm
  o LSTM
### 3.3 Interfaces
o Input and output data are extracted from stock market website
o Streamlit application is used for deployment
o Syntax error and logical errors are taken into consideration
### 3.4 Error Handling
The model handled getting inputs, numerical errors, model loading, and data transformations with
separate exception handlers.
### 3.5 Performance
Expected response times
o The system logged every event so that the user knows which process is running internally.
o The system identifies at which step logging required.
o The system logged each and every system flow
### 3.6 Resource usage
When a task is performed, it used all the processing power available until its work done.
## 4. Project Execution:
### 4.1 Data Export:
The accumulated data from database is exported in csv format for model training.
### 4.2 Data Loading:
o Raw data is imported and displayed in the application for the last 20 years.
o Chart is displayed with the available data
o Range slider is used to zoom in and zoom out the available data.
### 4.3 Data Transformation:
o Each data has been subjected to scaler transformation.
o Difference timing steps till 100 is transformed as 100 features.
### 4.4 Train Test Split:
Data are separated for training and testing purpose. For testing purpose 35% of data is used.
### 4.5 Model Training:
The models used for training are Long Short Term Memory and Facebook Prophet. Among these
methods LSTM performed well.
### 4.6 Performance Evaluation:
The model's performance was assessed using accuracy score. An RMSE of 22.39 is achieved via the
Long Short Term Memory algorithm.
## 5. Deployment:
The model is deployed using Stremlit Application. The application runs on the local host effectively.
## 6. Conclusion:
In conclusion, this investment prediction project utilized machine learning techniques to analyze
historical stock market data and predict future trends. Our analysis focused on the technology sector,
and we used a combination of time series analysis and LSTM to build a model that accurately predicted
future stock prices.
Our findings revealed several promising investment opportunities in the technology sector, with
particular emphasis on companies in any sector. Our model was able to accurately predict the stock
prices of example companies, with an RMSE 22.39.
However, we also identified some potential risks associated with investing in the technology sector,
including market volatility and the risk of overvaluation. Investors should carefully consider these risks
before making any investment decisions.
In future work, we suggest exploring the use of additional features, such as social media sentiment
analysis and news sentiment analysis, to improve the accuracy of the model.
Overall, this project highlights the potential benefits of using machine learning techniques to inform
investment decisions. Our findings suggest that with careful analysis and consideration of the risks, the
technology sector offers promising investment opportunities for investors.
