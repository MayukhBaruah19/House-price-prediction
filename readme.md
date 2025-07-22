# House Price Prediction : 
House price prediction is a machine learning task that involves estimating the selling price of residential properties based on various features such as location, size, number of rooms. It is a type of regression problem where the goal is to build a model that can learn from historical housing data and make accurate predictions on new, unseen data. This type of prediction is highly valuable in the real estate industry, helping buyers, sellers, investors, and financial institutions make informed decisions. By analyzing trends and relationships within the data, predictive models can provide insights into market behavior and support more efficient and data-driven property valuations.<br>
![image](https://www.appliedaicourse.com/blog/wp-content/uploads/2025/01/House-Price-Prediction-Using-Machine-Learning.png)

# About this project :
This project aims to predict house
prices in Guwahati using a machine learning models, I have use several machine learning algorithm such as Linear Regression, RandomForestRegressor, DecisionTreeRegressor, Support Vector Machine etc but specifically used Linear Regression model. Linear
regression is a fundamental and widely used algorithm
for predictive modeling, particularly in cases where the
relationship between the dependent variable (house
prices) and independent variables (factors such as size,
location, number of rooms, etc.) is assumed to be
linear, The choice of linear regression is ideal for this
project due to its simplicity and interpretability.To enhance user interaction, a Graphical User Interface (GUI) was developed using Streamlit, allowing users to input property features and receive real-time price predictions in an intuitive and user-friendly environment.
# Demo :
![image](https://github.com/MayukhBaruah19/House-price-prediction/blob/main/templets/Screenshot%202025-07-22%20225243.png)<br>
![image](https://github.com/MayukhBaruah19/House-price-prediction/blob/main/templets/Screenshot%202025-07-22%20225454.png)
# DataSet Used : 
- [Dataset](https://www.kaggle.com/datasets/reckonmazumdar/guwahati-house-price-data)
# How to run?  
### STEP 1 :
Clone the repository
```Bash
https://github.com/MayukhBaruah19/House-price-prediction.git
```
### STEP 2 : Create a conda environment after opening the repository
```Bash
conda create -p venv python=3.12 -y
```
```Bash
conda activate venv
```
### STEP 3 : install the requirements
```Bash
pip install -r requirements.txt
```
### STEP 4 : Run the ```main.py``` file
```Bash
streamlit run main.py
```

# Project Structure
```Bash                    
├── notebooks/
|     ├──Fraud_Detection_Model.ipynb       # Jupyter notebooks for model development
|     ├──Data/
|          └── guwahati_house_price.csv    # Data       
|            
├── artifacts/                             # .pkl files importent for the project 
|     └── onehot_encoder_location.pkl
|     └──regression_model.pkl
|     └── scaler.pkl
├──main.py                                 # Source code for GUI
├── README.md                              # Project documentation
└── requirements.txt                       # Required packages
``` 
# Contributors :
- [Mayukh Baruah](https://www.linkedin.com/in/mayukh-baruah-528116290/)
