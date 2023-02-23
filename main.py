import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
pip install tensorflow
import tensorflow as tf
import streamlit as st
# from PIL import Image
# import requests
# from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Sample websapp")


@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/content/ann_youtubeadview.h5')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()


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
