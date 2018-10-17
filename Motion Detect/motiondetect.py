import sys

import numpy       # importing Numpy for use w/ OpenCV
import cv2                            # importing Python OpenCV
from datetime import datetime         # importing datetime for naming files w/ timestamp
import time
import os

from detect import detectLicensePlate, consulta_Listanegra

def diffImg(t0, t1, t2):              # Function to calculate difference between images.
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

def loadWeights(weights):
    f = numpy.load(weights)

    for ii in numpy.load(weights):
        if type(f[ii]) != numpy.ndarray:
            f.files.pop(f.files.index(ii))

    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[-1]))]
    return param_vals

if __name__ == "__main__":

    threshold = 1000000             # Threshold for triggering "motion detection"
    cam = cv2.VideoCapture('../Motion Detect/videos/IMG_6720.MOV')             # Lets initialize capture on webcam

    #cv2.namedWindow("Movement Indicator")              # comment to hide window
    # Read three images first:
    t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    # Lets use a time check so we only take 1 pic per sec
    timeCheck = datetime.now().strftime('%Ss')

    pv = loadWeights(sys.argv[1])

    print("\ndetect start", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    while cam.isOpened():
        ret, frame = cam.read()	      # read from camera
        if ret:
            totalDiff = cv2.countNonZero(diffImg(t_minus, t, t_plus))	# this is total difference number
            text = "threshold: " + str(totalDiff)				# make a text showing total diff.
            #print (text)
            t_minus = t
            t = t_plus
            t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
            cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)   # display it on screen
            if totalDiff > threshold and timeCheck != datetime.now().strftime('%Ss'):
                dimg= cam.read()[1]
                crop_img = dimg[50:370, 100:1160]
                
                # cv2.imwrite(os.path.join('../Motion Detect/imgs detected/',datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg') , crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                # detectLicensePlate(crop_img, pv)
                # name_file = datetime.now().strftime('%Y%m%d_%Hh%M%Ss%f') + '.jpg' 
                name_file, code=detectLicensePlate(crop_img, pv)
                
                if len(name_file)>1:
                    cv2.imwrite(os.path.join('../Motion Detect/imgs detected/', name_file) , crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    consulta_Listanegra(code, name_file) # funcion para consultar a lista negra y enviar notificacion, params: id_vehiculo, placa, y ruta imagen
                    #cv2.imwrite(os.path.join('./imgs detected/', name_file) , crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                timeCheck = datetime.now().strftime('%Ss')
                # Read next image

            #cv2.imshow(winName, frame)
            key = cv2.waitKey(10)
            if key == 27:			 # comment this 'if' to hide window
                cv2.destroyWindow(winName)
                break
        else: break
    print("detect end", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
