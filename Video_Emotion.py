# For image processing
import cv2

# For loading images
import numpy as np
from keras.models import load_model

# Face detect model
net = cv2.dnn.readNetFromCaffe("./Models/deploy.prototxt.txt", "./Models/res10_300x300_ssd_iter_140000.caffemodel")
emotion_model_path = './Models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}

def detect_face(frame,confidence=0.4):

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



def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, -1)    
    return x


# In[26]:


# Return landmarks as numpy array
def np_emotion(frame, verbose=False):
    #         Detect the face
        face_locations = detect_face(frame)
        (face_start_x, face_start_y, face_end_x, face_end_y) = face_locations
        
        gray_image = cv2.cvtColor(frame[face_start_x: face_end_x,face_start_y:face_end_y], cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_image, (emotion_target_size))
        gray_face = preprocess_input(gray_face)
        emotion = emotion_classifier.predict(gray_face)
        
        if verbose == True:
            cv2.rectangle(frame, (face_start_x, face_start_y ), (face_end_x, face_end_y ),(255,0,0),2)
        
        return emotion,face_end_x, face_end_y


# # Load our MLP

# In[27]:


# Run webcam, for later testing
def webcam_run():
    video_capture = cv2.VideoCapture(0)

    while(True):
        ret, frame = video_capture.read()
        try:
            emotion,x,y = np_emotion(frame,True)
            emotion = labels[np.argmax(emotion)] 
            
            cv2.putText(frame, emotion, (x-60, y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            pass
    video_capture.release()
    cv2.destroyAllWindows()
webcam_run()

