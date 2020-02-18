# !pip install opencv-python
import cv2
# Define Capture Function
points=[]

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        points.append([x,y])

img = cv2.imread('jpg/ct.jpg')
cv2.imshow('image', img)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
print(points)
