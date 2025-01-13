import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import csv
import joblib
import time
import random

############# GAME STATE VARIABLES ###########################
playerScore = 0
aiScore = 0

playerSign = ""
aiSign = ""
winner = ""
isStarted = False
timerRunning = False
initialTime = 0
previousTime = 0

############# TRAINING VARIABLES #####################
isTraining = False
trainingSlot = '0'

################# HELPER METHODS ###############
def normalise(landmarks, boundingBox):
    normalised_Lm = []
    x_min, y_min, box_width, box_height = boundingBox
    
    for lm in landmarks:
        x, y, z = lm
        norm_x = (x - x_min) / box_width
        norm_y = (y - y_min) / box_height
        norm_z = z / max(box_width,box_height)
        normalised_Lm.append([norm_x, norm_y, norm_z])
    
    return normalised_Lm

def determineWinner(ai, player):
    if player == ai:
        return "draw"
    elif player == "rock":
        if ai == "scissors" or ai == "lizard":
            return "player"
        else:
            return "ai"
    elif player == "paper":
        if ai == "rock" or ai == "spock":
            return "player"
        else:
            return "ai"
    elif player == "scissors":
        if ai == "paper" or ai == "lizard":
            return "player"
        else:
            return "ai"
    elif player == "lizard":
        if ai == "spock" or ai == "paper":
            return "player"
        else:
            return "ai"
    elif player == "spock":
        if ai == "rock" or ai == "scissors":
            return "player"
        else:
            return "ai"
    else:
        return "ai"
        
def updateScores(winner):
    global playerScore
    global aiScore
    
    if winner == "player":
        playerScore += 1
    elif winner == "ai":
        aiScore += 1

############ LOAD MODEL ################################
labels = ["rock", "paper", "scissors", "lizard", "spock"]
clf = joblib.load('./model/model.pkl')

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
    
    ##### Flip camera accross the y-axis to mirror correctly ####
    cam = cv2.flip(cam, 1)
    
    ##### Crop camera img to fit inside player box on the bg image ########
    cam = cam[:,80:283]
        
    ###### get background img and resize ######
    imgBG = cv2.imread("UI/bg2.png")
    imgBG = cv2.resize(imgBG, (960, 540)) 
    
    ########## Update UI Scores ############
    cv2.putText(imgBG, str(playerScore), (770, 173), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(imgBG, str(aiScore), (182, 173), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) 
    
    ########## Display AI sign on UI ############
    cv2.putText(imgBG, aiSign.upper(), (240, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1, cv2.LINE_AA)
    if aiSign != "":
        imgAiSign = cv2.imread(f"./UI/{aiSign}.png", cv2.IMREAD_UNCHANGED)
        imgAiSign = cv2.resize(imgAiSign, (80, 80))
        imgBG = cvzone.overlayPNG(imgBG, imgAiSign, pos=[225,255])
    
    ########## Display Winner on UI ############
    if winner != "":
        result_text = ""
        if winner == "player":
            result_text = "You Win!"
        elif winner == "ai":
            result_text = "AI Wins!"
        else:
            result_text = "Draw!"
        cv2.putText(imgBG, result_text, (420, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 250), 2, cv2.LINE_AA)
    
    ########## Find hands and update image ######
    hands, cam = detector.findHands(cam, draw=True, flipType=False)

    ###### Detect hand sign using model ##############
    if hands:     
        
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        bbox = hand['bbox']
        x_min, y_min, box_width, box_height = bbox
        normalised_lm = normalise(hand['lmList'], bbox)
        flattened_lm = [item for sublist in normalised_lm for item in sublist]
        
        # Query the model
        prediction = clf.predict([flattened_lm])
        playerSign = labels[prediction[0]]
        
        # Display the player sign on UI
        cv2.putText(imgBG, playerSign.upper(), (620, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1, cv2.LINE_AA)
        # print(f'Detected sign: {playerSign}')
    else:
        playerSign = ""
    
    
    if timerRunning:
        timer = time.time()- initialTime
        timer = 3 - int(timer)
        if timer != previousTime:
            previousTime = timer
            print("Time left:", timer)
        
        cv2.putText(imgBG, str(timer), (465, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 250, 250), 2, cv2.LINE_AA)
        
        if timer <= 0:
            # print("Time's up!")        
            #generate ai sign
            randomIndex = random.randint(0,4)
            aiSign = labels[randomIndex]
            # print("AI sign:", aiSign)
            # print("Player sign:", playerSign)

            #determine winner
            winner = determineWinner(aiSign, playerSign)
            print("Winner:", winner)
            
            updateScores(winner)
            print("Player Score:", playerScore)
            print("AI Score:", aiScore)
        
            timerRunning = False
            isStarted = False

            
            
    
    
    
    
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
    
    elif key == ord('c') and isTraining and hands: #capture hand data and append to training data file
        hand = hands[0]
        print("Hand data captured")
        print("Label:", trainingSlot)
        print("Handedness:", hand['type'])
        landmarks = normalise(hand['lmList'],hand['bbox'])
        print("Hand Landmarks (normalised):", landmarks)
        
        with open ('./model/training/training_data.csv', mode="a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([trainingSlot, hand['type'], landmarks])
    
    #start game
    elif key == ord('s') and (not isStarted):
        print("Game started")
        aiSign = ""
        winner = ""
        isStarted = True
        timerRunning = True
        initialTime = time.time()
        
        
            
    #end program        
    elif key == ord('q'):
          break



    