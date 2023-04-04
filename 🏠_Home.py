import streamlit as st
from PIL import Image

st.set_page_config(page_title = 'Movie Recommendation System', page_icon = '🎥')
st.markdown("<h1 style='text-align: center; color: grey; font-family: Courier;'>Movie Recommendation System</h1>", unsafe_allow_html=True)
for _ in range(5):
    st.write(' ')
image = Image.open('movie-recommendation.jpg')
st.image(image, use_column_width = True)
