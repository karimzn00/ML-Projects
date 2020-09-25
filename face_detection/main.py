import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



def detect(gray, frame):
	faces = face_cascade.detectMultiScale(gray, 1.2, 7, minSize = (30, 30))
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
	return frame

cam = cv2.VideoCapture(0)
while True:
  _, frame = cam.read()
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = detect(gray, frame)
  cv2.imshow('Face Detection', faces)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cam.release()
cv2.destroyAllWindows()