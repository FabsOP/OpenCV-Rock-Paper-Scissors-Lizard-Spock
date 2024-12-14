import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

################## SET UP WEBCAM ######################
camera = cv2.VideoCapture(0)

camera.set(3,960)                        #camera width
camera.set(4,540)                        #camera height

############ CREATE HAND DETECTOR #######################
detector = HandDetector(maxHands=1)

############ FRAME UPDATE LOOP ##########################
while True:
    ##### get camera img ##########
    success, cam = camera.read()       #return success boolean and image
    
    ########## resize camera img 205/540 ##########
    cam = cv2.resize(cam, (0,0), None, 0.379, 0.379)
    
    ##### Flip camera accross the y-axis to mirror user correctly ####
    cam = cv2.flip(cam, 1)
    
    ##### Crop camera img to fix in player box ########
    cam = cam[:,80:283]
        
    ###### get background img and resize ######
    imgBG = cv2.imread("UI/bg.png")
    imgBG = cv2.resize(imgBG, (960, 540)) 
    
    ########## Find hands and update image ######
    hands, cam = detector.findHands(cam, draw=True, flipType=False)
    
    
    ########## Place Camera img on bg img ###############
    imgBG[199:404,598:801] = cam
    
    ########## Show images ####################
    #Show camera img
    # cv2.imshow("You", cam)
    
    #show background img
    cv2.imshow("bg", imgBG)
    
    ################# End of Loop ###########
    #refresh frequency
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break