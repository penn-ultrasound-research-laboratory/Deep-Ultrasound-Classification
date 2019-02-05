import cv2
import os

from src.utilities.segmentation.xianauto.automatic import get_ROI

for i, f in enumerate(os.listdir("../TestImages/bank")):
    
    img = cv2.imread("../TestImages/bank/{}".format(f), cv2.IMREAD_GRAYSCALE)
    roi = get_ROI(img)
            
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)