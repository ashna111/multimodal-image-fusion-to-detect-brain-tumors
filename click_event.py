# !pip install opencv-python
import cv2
# Define Capture Function
points=[]
def capture_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: #left button of the mouse
            print(x,y)
            points.append = ((x, y))
            
# Load Image
image = cv2.imread('jpg/ct.jpg')
cv2.setMouseCallback("image", capture_click)

while True:
    cv2.imshow("Image", image)
    key=cv2.waitKey(1)
    if key==27:
        break

# cv2.waitKey(0) #press any key to close all windows
cv2.destroyAllWindows()
points