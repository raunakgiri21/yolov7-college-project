import cv2
import time
import requests
import random
import numpy as np
import yaml
from yaml.loader import SafeLoader
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple


class YOLO_Pred():

    def __init__(self,w,data_yaml,session):
        # load YAML
        with open(data_yaml,mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

        self.w = w
        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        self.session = session

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def predictions(self,img):
        names = self.labels
        random.seed(10)
        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        outname = [i.name for i in self.session.get_outputs()]

        inname = [i.name for i in self.session.get_inputs()]

        inp = {inname[0]:im}

        # ONNX inference
        outputs = self.session.run(outname, inp)[0]

        ori_images = [img.copy()]
        countArr = [0]*self.nc
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            countArr[cls_id]+=1
            score = round(float(score),3)
            # print(score)
            name = names[cls_id]
            color = colors[name]
            name += ' '+str(score)
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.5,[225, 255, 255],thickness=2)

        countDict = self.getCounts(countArr)    
        return {'pred_img': ori_images[0], 'count_dict': countDict}

    def getCounts(self,countArr):

        myDict = {'Auto': 0,
            'Bus': 0,
            'Car': 0,
            'Lcv': 0,
            'Motorcycle': 0,
            'Multiaxle': 0,
            'Tractor': 0,
            'Truck': 0 }
        # print("\n\n\nTotal number of vehicles in this frame: ",sum(countArr),"\n")
        for i in range(self.nc):
            myDict[(self.labels[i]).capitalize()] = countArr[i]
            # print(self.labels[i]+": ",countArr[i])
        return myDict    


        