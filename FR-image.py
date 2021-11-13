import numpy as np
import cv2
from random import randrange


#load some pretrained data on face frontals from opencv (haarcascade algorithm) 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#choosing an image to detect faces in
img = cv2.imread('revahack.jpg')


#must convert to grayscale
grayscale_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    
#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    
#draw rectangles around faces
for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w , y+h), (0,255,0), 2)
    
#displaying the image with detection
cv2.imshow('Clever Face Detector',img)
cv2.waitKey()


print("Code Completed!")


 