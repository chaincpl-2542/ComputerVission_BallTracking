import cv2
import numpy as np

lower_color = np.array([29, 86, 6])  
upper_color = np.array([64, 255, 255]) 

video_path = "ball_tracking_example.mp4"  
cap = cv2.VideoCapture(video_path)


kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500: 
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:  
        print("ESC pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
