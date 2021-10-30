import cv2
from playsound import playsound

body_classifire=cv2.CascadeClassifier("haarcascade_fullbody.xml")
cap = cv2.VideoCapture("vtest.avi")

while cap.isOpened():
    ret_, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies=body_classifire.detectMultiScale(gray,1.1,3)

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        cv2.putText(gray, "Intrusion Detected", (210, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        #playsound('alarmsound.mp3')
        cv2.imshow("Intrusion Detection System", frame)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()