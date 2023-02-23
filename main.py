import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import keras
import pandas as pd
import tensorflow as tf
import streamlit as st

from prophet.plot import *
from prophet.serialize import model_to_json, model_from_json
# from PIL import Image
# import requests
# from io import BytesIO


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Sample websapp")

cities = ('Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal')
selected_city = st.selectbox('Select dataset for prediction', cities)

@st.cache(allow_output_mutation=True)
def load_model():
  with open('content/serialized_model.json', 'r') as fin:
    m = model_from_json(fin.read())  # Load model
  return m

with st.spinner('Loading Model Into Memory....'):
  m= load_model()

data_load_state = st.text('Loading data...')

future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)


st.subheader('predicted data')
st.write(forecast.tail())

st.subheader('graph')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# def plot_raw_data(data):
# 	fig = go.Figure()
# 	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
# 	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
# 	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
# 	st.plotly_chart(fig)
	
# plot_raw_data(prediction)

# def decode_img(image):
#   img = tf.image.decode_jpeg(image, channels=3)  
#   img = tf.image.resize(img,[224,224])
#   return np.expand_dims(img, axis=0)

# path = st.text_input('Enter Image URL to Classify.. ','https://beanipm.pbgworks.org/sites/pbg-beanipm7/files/styles/picture_custom_user_wide_1x/public/AngularLeafSpotFig1a.jpg')
# if path is not None:
#     content = requests.get(path).content

#     st.write("Predicted Class :")
#     with st.spinner('classifying.....'):
#       label =np.argmax(model.predict(decode_img(content)),axis=1)
#       st.write(classes[label[0]])    
#     st.write("")
#     image = Image.open(BytesIO(content))
#     st.image(image, caption='Classifying Bean Image', use_column_width=True)




