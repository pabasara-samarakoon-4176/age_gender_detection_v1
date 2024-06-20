import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import streamlit as st
import PIL
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# gender_model = tf.keras.models.load_model('model/train_gender_model.h5')
# age_model = tf.keras.models.load_model('model/train_age_model.h5')
def create_gender_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('model/train_gender_model.h5')
    return model

def create_age_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.load_weights('model/train_age_model.h5')
    return model

gender_model = create_gender_model()
age_model = create_age_model()

def predictor(file):
    img = PIL.Image.open(file)
    width, height = img.size
    if width == height:
        img = img.resize((200, 200), PIL.Image.LANCZOS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            img = img.crop((left, top, right, bottom))
            img = img.resize((200, 200), PIL.Image.LANCZOS)
        else:
            left = 0
            right = width 
            top = 0
            bottom = width 
            img = img.crop((left, top, right, bottom))
            img = img.resize((200, 200), PIL.Image.LANCZOS)
            
    ar = np.asarray(img)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 200, 200, 3)
    
    age = age_model.predict(ar)
    gender = np.round(gender_model.predict(ar))
    if gender == 0:
        gender = 'male'
    elif gender == 1:
        gender = 'female'
        
    return img.resize((300, 300), PIL.Image.LANCZOS), int(age), gender

st.title("Age and gender detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    resized_img, predicted_age, predicted_gender = predictor(uploaded_file)

    st.image(resized_img, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Age: {predicted_age}")
    st.write(f"Predicted Gender: {predicted_gender}")

