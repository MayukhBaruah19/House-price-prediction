import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from PIL import Image


model = pickle.load(open('model_forest.pkl', 'rb'))
encoder = pickle.load(open('le_encode.pkl', 'rb'))
StandardScaler = pickle.load(open('scaled_data.pkl', 'rb'))

st.title("Guwahati House price prediction")
st.sidebar.header("Put your Data :")


def user_data():
    Bhk = st.sidebar.slider('No of Bhk', 0, 10, 3)
    locations = (
        'CHRISTIAN BASTI', 'LAL GANESH', 'BORAGAON', 'BAGHARBARI',
        'BELTOLA', 'AHOM GAON', 'BAMUNIMAIDAM', 'KAHILIPARA', 'SIX MILE',
        'REHABARI', 'LOKHRA', 'JYOTIKUCHI', 'SARANIA HILLS', 'GANESHGURI',
        'SACHAL PATH VIP ROAD BYLANE NUMBER 1', 'CHANDMARI', 'NOONMATI',
        'ULUBARI', 'DISPUR', 'SARUMOTORIA', 'SOUTH SARANIA ROAD',
        'DOWNTOWN', 'JATIA', 'ATHGAON', 'DHARAPUR', 'RADHA NAGAR',
        'BAMUNIMAIDAN', 'GHORAMARA', 'ZOO TINIALI', 'BORBARI',
        'KAHILIPARA ROAD', 'ZOO ROAD', 'AZARA', 'PATOR KUCHI', 'BASISTHA',
        'KALA PAHAR', 'LALMATI', 'RUKMINI GAON', 'MATHGHARIA', 'KHANAPARA',
        'BHANGAGARH', 'GEETANAGAR', 'SARUSAJAI', 'BHETAPARA', 'HATIGAON',
        'MALIGAON', 'LACHIT NAGAR', 'HENGRABARI', 'NAYANPUR',
        'NARENGI TINALI', 'SATGAON', 'JALUKBARI', 'BARSAPARA',
        'PANJABARI ROAD', 'ADABARI', 'GS ROAD', 'VIP ROAD', 'GARCHUK',
        'JAYANAGAR', 'KALYANI SAGAR PATH', 'DIGHALIPUKHURI', 'KERAKUCHI',
        'SAWKUCHI', 'ABC GALI', 'TARUN NAGAR', 'PANJABARI BUS STAND',
        'BEHARBARI CHARIALI', 'KAMAKHYA', 'UZAN BAZAR', 'CHACHAL ROAD',
        'BHETAPARA GHORAMARA ROAD', 'BETKUCHI', 'SIJUBARI', 'PANJABARI',
        'SOUTH SARANIA', 'RUPNAGAR', 'GUWAHATI', 'DHIRENPARA',
        'PALTAN BAZAAR', 'SURVEY', 'NALAPARA', 'LOKHRA ROAD', 'MANPARA',
        'SILPUKHURI', 'PANDU', 'SUNDARBARI', 'CHANDAN NAGAR', 'LANKESHWAR',
        'PATHAR QUARRY', 'KAHIKUCHI', 'KAHILPARA', 'NARIKAL BARI',
        'BIRUBARI', 'GANDHI BASTI', 'KHARGHULI HILLS', 'BASISHTA',
        'SARABBHATI', 'DWARAKA NAGAR', '9TH MILE', 'RAJGARH ROAD',
        'PRAGJYOTISH NAGAR', 'LAKHIMI NAGAR', 'AMBIKAGIRINAGAR', 'TETELIA',
        'BHARALUMUKH', 'PAN BAZAR', 'DATALPARA', 'ADAGUDAM', 'JAPORIGOG',
    )

    location = st.sidebar.selectbox("location", locations)
    Size = st.sidebar.slider(
        "Size of the house (in square feet)", 0, 7800, 500)

    ok = st.button("Calculate price")
    if ok:
        op = np.array([[Size, Bhk, location]])
        op[:, 2] = encoder.transform(op[:, 2])
        price = model.predict(op)
        st.subheader(f"The estimated price is ₹{price[0]:.2f} Lakhs")


user_data()
