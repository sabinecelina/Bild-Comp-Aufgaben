import numpy as np
import cv2 

# testprint
print("hello world")

cap = cv2.VideoCapture(0)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print("height : ", height)
print("width: ", width)

while(True):
    img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    # ret is a boolean that returns true if the frame is available.
    # frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
    if(ret):    
        print("yes")
    if cv2.waitKey(10) == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# # Destroy all the windows
cv2.destroyAllWindows()

# TODO release the video capture object and window

