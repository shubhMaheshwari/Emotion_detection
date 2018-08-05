import numpy as np
import sys
try:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
	import cv2
	from imutils import face_utils
	print("[ INFO] OPENCV-VERSION:",format(cv2.__version__))	
	sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')	
except Exception as e:
	print(e)
	import cv2
	pass
import dlib
import os


print("[ INFO] loading model for face detection...")
CURRENT_PWD = os.path.join(os.getcwd(),"Models")
net = cv2.dnn.readNetFromCaffe(os.path.join(CURRENT_PWD,"deploy.prototxt.txt"), os.path.join(CURRENT_PWD,"res10_300x300_ssd_iter_140000.caffemodel"))
detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')


def detect_face_SSD(frame,confidence=0.4):

	(h, w) = frame.shape[:2]
	if h == 0 or w == 0:
		return [] 
	try:
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
	except:
		return []

	net.setInput(blob)
	detections = net.forward()
	detections = detections[0,0,:,:]
	detections = detections[detections[:,2] > confidence]
	
	face_locations = ()
	for f in detections:
		(face_startX, face_startY, face_endX, face_endY)  = (f[3:7]*np.array([w,h,w,h])).astype(int)
		face_locations = (face_startX , face_startY , face_endX , face_endY)

	return face_locations

def detect_face_dlib(frame):
	(h, w) = frame.shape[:2]
	if h == 0 or w == 0:
		return [] 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)

	face_locations = ()

	for rect in rects:
		(x, y, wid, hei) = face_utils.rect_to_bb(rect)
		face_locations = (x , y , x + wid , y + hei )

	return face_locations

def detect_face_haar(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if faces is not ():
		(x, y, wid, hei) = faces[0]	
		return (x , y , x + wid , y + hei )
	else: 
		return () 

def detect_face(frame,method=None):
	if method == "dlib":
		return detect_face_dlib(frame)
	elif method == "haar":
		return detect_face_haar(frame)
	else :
		return detect_face_SSD(frame)