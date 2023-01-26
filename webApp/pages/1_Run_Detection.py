import streamlit as st
from PIL import Image
import numpy as np
st.set_page_config(page_title="Detection", layout='wide', page_icon='./webApp/images/object.png')
st.title('Welcome to Detection Page')
st.caption('Try object detection by uploading an image!')




# ................ Loading Model...............
import onnxruntime as ort
from time import sleep
from "webApp/yolo_predictions" import YOLO_Pred

modelSuccessStatus = ''

with st.spinner("Preparing Your Yolo Model..."):
    try:
        w = './webApp/models/best.onnx'
        cuda = True
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = ort.InferenceSession(w, providers=providers)

        yolo = YOLO_Pred(w,'./webApp/models/data.yaml',session)
        sleep(0.5)
        modelSuccessStatus = st.success("Your Model is Active!")
    except:
        sleep(0.25)
        modelSuccessStatus = st.error("Unable to Activate YOLO Model")    





# ................ Load Image ..................
st.write('Please Upload an Image to get detections')

def uploadFile():
    image_file = st.file_uploader(label='Upload Image', accept_multiple_files= False)
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": f"{round(size_mb,2)} MB",
        }
        # st.json(file_details)
        # validate filetype
        if not (file_details['filetype'] in ('image/png','image/jpeg')):
            st.error("Uploaded file is not an image/frame.")
            return None
        else:
            modelSuccessStatus.empty()
            st.success("File Uploaded Succesfully!")
            return {"file": image_file, "fileDetails": file_details}




# ................ Main Function ................

def main():
    myObject = uploadFile()
    if myObject:
        image_obj = Image.open(myObject['file'])
        image_obj = image_obj.resize((640,640))
        button = None
        # frame = cv2.resize(myObject,(640,640),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        # pred_image = yolo.predictions(frame)
        col11, col12 = st.columns(2)
        col21, col22 = st.columns(2)
        with col11:
            st.info("Preview of the image")
            st.image(image_obj)
        with col21:    
            st.subheader("File Details:")
            st.json(myObject['fileDetails'])
            button = st.button("Run Detections")       
        with col12:
            if button:
                image_array = np.array(image_obj)
                pred_array = yolo.predictions(image_array)['pred_img']
                pred_img = Image.fromarray(pred_array)
                col12 = st.empty()
                st.info("Detections in the image")
                st.image(pred_img)
        with col22:
            if button:
                image_array = np.array(image_obj)
                count_dict = yolo.predictions(image_array)['count_dict']
                st.subheader("Vehicle Counts:")
                st.write("Total Vehicles detected:  ***{}***".format(sum(count_dict.values())))
                st.json(count_dict)      

        

if __name__ == "__main__":
    main()