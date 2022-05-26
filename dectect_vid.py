# To run this file:
'''
    pip install cv2
    pip install deepface
    pip install retina-face
    put the video in the same folder with this file
'''

import cv2
from deepface import DeepFace
from retinaface import RetinaFace
import numpy as np
import pandas as pd
from numpy import mean
import matplotlib.pyplot as plt

def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes

# Load the cascade  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# To capture video from existing video.   
cap = cv2.VideoCapture('Video Of People Walking.mp4')  
# cap = cv2.VideoCapture(0)

i = 0
while True:  
    # Read the frame  
    #_, img = cap.read()
    ret, frame = cap.read()

    bbox_array = np.zeros([480,640,4], dtype=np.uint8)

    cv2.imshow('Video', frame) 
    print('helo')
    
    #cv2.rectangle(frame, (200,100),(300,400), (255,255,255), 2) # BGR

    cv2.imwrite('bbox{}'.format(i), frame)
    i+=1
    # def Detect_Retina
    # Face detect using Retinaface  
    # obj = RetinaFace.detect_faces(img)
  
    # print('='*100,'\n'
    #   'found', len(obj.keys()), 'face in picture','\n',
    #    '='*100)
    # i = 1
    # for key in obj.keys():
  
    #     identity = obj[key]

        # r_eye = identity['landmarks']['right_eye']
        # l_eye = identity['landmarks']['left_eye']
        # nose = identity['landmarks']['nose']
        # mouth_left = identity['landmarks']['mouth_left']
        # mouth_right = identity['landmarks']['mouth_right']

        # # detect and draw line on eye, nose, mouth 
        
        # cv2.line(img, (int(r_eye[0]),int(r_eye[1])), (int(l_eye[0]),int(l_eye[1])), (0,255,0), 2)
        # cv2.line(img, (int(r_eye[0]),int(r_eye[1])), (int(nose[0]),int(nose[1])), (0,0,255), 2)
        # cv2.line(img, (int(l_eye[0]),int(l_eye[1])), (int(nose[0]),int(nose[1])), (0,0,255), 2)
        # x = (int(mouth_left[0]) + int(mouth_right[0]))/2
        # y = (int(mouth_left[1]) + int(mouth_right[1]))/2
        
        # cv2.line(img, (int(mouth_left[0]),int(mouth_left[1])), (int(mouth_right[0]),int(mouth_right[1])), (0,0,255), 2)
        # cv2.line(img, (int(nose[0]),int(nose[1])), (int(x),int(y)), (255,0,0), 2)

        # rectangle faces
    
        # facial_area = identity['facial_area']
        # cv2.rectangle(img2, (facial_area[0],facial_area[1]),(facial_area[2],facial_area[3]), (255,255,255), 2) # BGR

    

        # text_size = (facial_area[2] - facial_area[0])/140
        # write up face
        #print((facial_area[2] - facial_area[0],facial_area[3] - facial_area[1]))
        # cv2.putText(img, 'Person.{}'.format(i) ,(facial_area[0],facial_area[1]- 15)
        # ,fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = text_size, color = (0,0,255))
        # i += 1

    # -----------------------------------------------------
    # Detect using opencv
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
    #   # Detect the faces  
    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
  
    # # Draw the rectangle around each face  
    # for (x, y, w, h) in faces:  
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #------------------------------------------------------
    # Face Recognition
    # person_name = 'Person'
    #   #print('img[',facial_area[1],':',facial_area[3],',',facial_area[0],':',facial_area[2],']')
    # df = DeepFace.find(img_path = img[facial_area[1]:facial_area[3],facial_area[0]:facial_area[2]]
    #                     , db_path =  path, detector_backend= 'mtcnn')
    # person_name = str(df['identity'][0].replace(path,'')).replace('.jpg','') 

    # cv2.putText(bbox_array, person_name,(facial_area[0],facial_area[1] - 12),
    #                 fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.8, color = (250,225,100))

    # Display  
    # cv2.imshow('Video', img2)
  
    # Press Esc to stop the video  
    k = cv2.waitKey(30) & 0xff  
    if k==27:  
        break  
          
# Release the VideoCapture object  
cap.release()  
cv2.destroyAllWindows()