import cv2
from deepface import DeepFace
from retinaface import RetinaFace


cam = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#faceCascade = DeepFace.detectFace(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#print(faceCascade)

while True:

    ret, frame = cam.read() 
    print(ret)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    #obj = RetinaFace.detect_faces(frame)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#    for key in obj.keys():
#        identity = obj[key]
#        #print(identity)

        #facial_area = identity['facial_area']
        #cv2.rectangle(frame, (facial_area[0],facial_area[1]),(facial_area[2],facial_area[3]), (0,0,255), 2) # BGR

    font = cv2.FONT_HERSHEY_SIMPLEX

    #emotRes = DeepFace.analyze(frame, actions = ['emotion'])
    # cv2.putText(frame, emotRes['emotion'], (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)
    #print(emotRes)

    cv2.putText(frame, "FPS: {}".format(cam.get(cv2.CAP_PROP_FPS)), (10,30), font, 1, (0, 255, 255), 2, cv2.LINE_4) 
    cv2.putText(frame, 'Press q to quit', (200,450), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow("ERS Visual V1.0 - Emotion Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
quit()