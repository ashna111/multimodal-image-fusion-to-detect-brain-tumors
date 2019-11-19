# !pip install opencv-python
import cv2
# Define Capture Function
points=[]
# def capture_click(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN: #left button of the mouse
#             print(x,y)
#             points.append = ((x, y))
            
# # Load Image
# image = cv2.imread('jpg/ct.jpg')
# cv2.setMouseCallback("image", capture_click)

# while True:
#     cv2.imshow("Image", image)
#     key=cv2.waitKey(1)
#     if key==27:
#         break

# # cv2.waitKey(0) #press any key to close all windows
# cv2.destroyAllWindows()
# points



# import numpy as np
# import cv2

#events = [i for i in dir(cv2) if 'EVENT' in  i]
#print(events)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,', ' ,y)
        points.append([x,y])
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # strXY = str(x) + ', '+ str(y)
        # cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
        # cv2.imshow('image', img)
    # if event == cv2.EVENT_RBUTTONDOWN:
    #     blue = img[y, x, 0]
    #     green = img[y, x, 1]
    #     red = img[y, x, 2]
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     strBGR = str(blue) + ', '+ str(green)+ ', '+ str(red)
    #     cv2.putText(img, strBGR, (x, y), font, .5, (0, 255, 255), 2)
    #     cv2.imshow('image', img)

#img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread('jpg/ct.jpg')
cv2.imshow('image', img)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
print(points)
