import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
# from model import KeyPointClassifier

############# UI VARIABLES ###########################
playerScore = 0
aiScore = 0

playerSign = ""
aiSign = ""

isTraining = False
trainingSlot = '0'

################## SET UP WEBCAM ######################
camera = cv2.VideoCapture(0)

camera.set(3,960)                        #camera width
camera.set(4,540)                        #camera height

############ CREATE HAND DETECTOR #######################
detector = HandDetector(maxHands=1)

############ LOAD MODEL ################################
labels = ["rock", "paper", "scissors", "lizard", "spock"]
# keypoint_classifier = KeyPointClassifier()



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

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        bbox = hand['bbox']
        x_min, y_min, box_width, box_height = bbox
        print('x_Min =', x_min)
        print('y_Min =', y_min)
        print('box_width =', box_width)
        print('box_height =', box_height)
        # print(hand['lmList'])
    
    
    ########## Place Camera img on bg img ###############
    imgBG[199:404,598:801] = cam
    
    ########## Show images ####################
    #Show camera img
    # cv2.imshow("You", cam)
    
    #show background img
    cv2.imshow("bg", imgBG)
    
    ################# Handle key presses ###########
    key = cv2.waitKey(1) & 0xFF
    
    #Toggle Training Mode 
    if key == ord('t'):  # Checks if 'T' is pressed
        print("Key 't' pressed")
        isTraining = not isTraining
        
        if isTraining:
            print("Training mode activated")
            print("(Training on slot #"+ trainingSlot+")")
        else:
            print("Training mode deactivated")
    
    #change training slot
    elif isTraining and key in list(map(ord,['0','1','2','3','4','5','6','7','8','9'])):
        trainingSlot = chr(key)
        print("Training slot #" + trainingSlot + " selected")
    
    #end program        
    elif key == ord('q'):
          break
    