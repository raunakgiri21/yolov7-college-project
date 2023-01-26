import streamlit as st
import os

st.set_page_config(page_title="Home", layout='wide', page_icon='./webApp/images/home.png')
st.title("Vehicle Classification")
st.caption('This web application demonstrate object detection using Yolo')
st.markdown("""
##### ***Example -***
""")

st.image('./webApp/images/sample_home.png')

# Content
st.markdown("""
### This App detects objects from images
- [Click here for App](/Run_Detection)
- Automatically detects ***8 different types of vehicles*** from image and also gives the ***Count*** of them.
##### Vehicle classifications are divided as below-
1. Auto
2. Bus
3. Car
4. Lcv
5. Motorcycle
6. Multiaxle
7. Tractor
8. Truck 
""")
