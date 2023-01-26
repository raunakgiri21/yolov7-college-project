import streamlit as st

st.set_page_config(page_title="About", layout='wide', page_icon='./webApp/images/about.png')
st.title('About This Project')
st.markdown("""
### What this project is?
- This project is made to detect the vehicle density in a particular road lane.
- The project is made using YOLOv7 custom dataset training.
- The datasets used in Yolo training is taken from Kaggle Open Soucre Datasets. [Copyrights](https://www.kaggle.com/datasets/sakshamjn/vehicle-detection-8-classes-object-detection)
- Our Model is trained with dataset based on cctv footage frames, so our model works better with cctv based test data.
*Note: This application only works with image based datasets*
### Why this project?
- Object detection is now a primary source of building Real Life Applications like weather predictions, injury detection, etc. Similarily, we have trained our model in such a way that it can detect the number of vehicles in a frame. We can detect whether the object is car, truck, bike, multiaxle vehicle, etc. Using the count of the vehicles, we can also build an algorithm to dynamically control the wait timer of traffic lights depending upon the vehicle densities of the lanes.
#### Check out our YOLO Model for Result and Analysis
- [Click here for Results](/Model_Analysis)
-------
##### Team: Group 01 [ETE'23, AEC]
""")