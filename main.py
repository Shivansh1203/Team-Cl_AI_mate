import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import keras
import pandas as pd
import tensorflow as tf
import streamlit as st
from plotly import graph_objs as go
# from PIL import Image
# import requests
# from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Sample websapp")

cities = ('Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal')
selected_city = st.selectbox('Select dataset for prediction', cities)

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('content/ann_youtubeadview.h5')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

data_load_state = st.text('Loading data...')

#testing part
data_test = pd.read_csv("content/test.csv") 
data_test=data_test[data_test.views!='F']
data_test=data_test[data_test.likes!='F']
data_test=data_test[data_test.dislikes!='F']
data_test=data_test[data_test.comment!='F']
category={'A': 1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
data_test["category"]=data_test["category"].map(category)

# Convert values to integers for views, likes, comments, dislikes and adview
data_test["views"] = pd.to_numeric(data_test["views"])
data_test["comment"] = pd.to_numeric(data_test["comment"])
data_test["likes"] = pd.to_numeric(data_test["likes"])
data_test["dislikes"] = pd.to_numeric(data_test["dislikes"])
column_vidid=data_test['vidid']

# Endoding features like Category, Duration, Vidid
from sklearn.preprocessing import LabelEncoder
data_test['duration']=LabelEncoder().fit_transform(data_test['duration'])
data_test['vidid']=LabelEncoder().fit_transform(data_test['vidid'])
data_test['published']=LabelEncoder().fit_transform(data_test['published'])

# Convert Time_in_sec for duration
import datetime
import time
def checki(x):
  y = x[2:]
  h = ''
  m = ''
  s = ''
  mm = ''
  P = ['H','M','S']
  for i in y:
    if i not in P:
      mm+=i
    else:
      if(i=="H"):
        h = mm
        mm = ''
      elif(i == "M"):
        m = mm
        mm = ''
      else:
        s = mm
        mm = ''
  if(h==''):
    h = '00'
  if(m == ''):
    m = '00'
  if(s==''):
    s='00'
  bp = h+':'+m+':'+s
  return bp

train=pd.read_csv("test.csv")
mp = pd.read_csv(path)["duration"]
time = mp.apply(checki)

def func_sec(time_string):
  h, m, s = time_string.split(':')
  return int(h) * 3600 + int(m) * 60 + int(s)

time1=time.apply(func_sec)

data_test["duration"]=time1
data_test=data_test.drop(["vidid"],axis=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_test = data_test
X_test=scaler.fit_transform(X_test)

prediction = model.predict(X_test)
prediction=pd.DataFrame(prediction)
prediction = prediction.rename(columns={0: "Adview"})



data_load_state.text('Loading data... done!')

st.subheader('predicted data')
st.write(prediction.tail())

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
