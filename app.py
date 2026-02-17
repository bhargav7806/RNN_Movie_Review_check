import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb 

model = load_model('simple_rnn_imdb.h5')

word_index = imdb.get_word_index()
reverse_word_index = {value : key for key , value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join(word_index.get (i - 3 , '$') for i in encoded_review)


def preprocess_text (text):
    words = text.lower().split()
    encoded_review = [word_index.get(word , 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review] , maxlen = 500)

    return padded_review



import streamlit as st

st.title('IMDB movie review analysis')
st.write('Enter a movie review')

user_input = st.text_area('Movie review')

if st.button('Click Here'):

    preprocess_input = preprocess_text(user_input)
    predicted = model.predict(preprocess_input)
    sentiment = 'postive' if predicted[0][0] > 0.5 else 'negative'

    st.write(f'Sentiment {sentiment}')
    st.write(f'prediction score {predicted[0] [0]}')

else:
    st.write('please enter a movie review ')

