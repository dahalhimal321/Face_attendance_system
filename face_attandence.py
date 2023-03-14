import face_recognition
import cv2
import os
from gtts import gTTS
import numpy as np
from playsound import playsound
from datetime import datetime
import csv

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
videocapture=cv2.VideoCapture(0)

saroj_image=face_recognition.load_image_file("C:/Users/ACER/Desktop/AI&ML projects/open_cv_faceattandence/photos/Saroj.jpg")
saroj_encoding=face_recognition.face_encodings(saroj_image)[0]

Himal_image=face_recognition.load_image_file("C:/Users/ACER/Desktop/AI&ML projects/open_cv_faceattandence/photos/himal.jpg")
Himal_encoding=face_recognition.face_encodings(Himal_image)[0]
Manu_image=face_recognition.load_image_file("C:/Users/ACER/Desktop/AI&ML projects/open_cv_faceattandence/photos/manu.jpg")
manu_encoding=face_recognition.face_encodings(Manu_image)[0]

Raju_image=face_recognition.load_image_file("C:/Users/ACER/Desktop/AI&ML projects/open_cv_faceattandence/photos/raju.jpg")
Raju_encoding=face_recognition.face_encodings(Raju_image)[0]




known_face_encoding=[Himal_encoding,saroj_encoding,manu_encoding,Raju_encoding]

known_face_names=["Himal","Saroj","Manu","Raju"]
students=known_face_names.copy()
face_locations=[]
face_encodings=[]
face_names=[]
s=True
now=datetime.now()
current_date=now.strftime("%Y-%m-%d")
f=open(current_date+'.csv','w+',newline='')
lnwriter=csv.writer(f)

while True:
    _,frame=videocapture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_name=[]
        for face_encoding in face_encodings:
            matches=face_recognition.face_distance(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]

                face_names.append(name)
                if name in known_face_names:
                    if name in students:
                     print(name)
                     
                     mytext='Thank you'
                     mytext1=mytext+name
                     language='en'
                     myobj=gTTS(text=mytext1,lang=language,slow=False)
                     myobj.save("thank.mp3")
                     file="thank.mp3"
                     playsound('thank.mp3')
                     os.remove(file)
                     
                     students.remove(name)
                     current_time=now.strftime("%H-%M-%S")
                     current_day=now.strftime("%A")

                     lnwriter.writerow([name,current_date,current_time,current_day])
                     
                else:
                    un='unknown person'
                    print(un)
                    lnwriter.writerow([un,current_date,current_time])
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
                
    
    
    
    cv2.imshow("attandence system",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
     break
videocapture.release()
cv2.destroyAllWindows()
f.close()


 
