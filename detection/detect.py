from email import header
import sys
import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Funtion to return angle between 3 points in R3

def ang(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    x = np.abs((np.arctan2(c[1]-b[1] , c[0]-b[0]) - np.arctan2(a[1]-b[1] , a[0] - b[0]))*180.0/np.pi)
    if x <= 180.0:
        return x
    else:
        return 360-x


# rsa = []
# rea = []
# rwa = []
# rpa = []
# rha = []
# rka = []
# raa = []
# lsa = []
# lea = []
# lwa = []
# lpa = []
# lha = []
# lka = []
# laa = []

headers = ['frame' , 'rea' , 'rwa' , 'rsa', 'rha' , 'rka' , 'lea' , 'lwa' , 'lsa', 'lha' , 'lka' ]
data = []
#data = [fNum , rElbowAng , rWristAng , rShoulderAng , rHipAng , rKneeAng , lElbowAng , lWristAng , lShoulderAng , lHipAng , lKneeAng]



cap = cv2.VideoCapture(sys.argv[1])
poseObj = mp_pose.Pose(min_detection_confidence=0.5 , min_tracking_confidence=0.5)

if cap.isOpened() == False:
    print("File not opened successfully")
    quit(1)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind(
    '/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_annotated.{inflext}'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

fNum = 0

while cap.isOpened() == True:

    success , img = cap.read()
    if not success:
        break

    fNum+=1
    #img = cv2.flip(img , 1)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = poseObj.process(img)
    

    

    #Storing Landmarks
    try:
        lm = results.pose_landmarks.landmark

        

        # Calculating Angles

        rShoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x , lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        rElbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x , lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        rWrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x , lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        rPinky = [lm[mp_pose.PoseLandmark.RIGHT_PINKY.value].x , lm[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
        rHip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x , lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        rKnee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x , lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        rAnkle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x , lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        

        lShoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x , lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        lElbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x , lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        lWrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x , lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        lPinky = [lm[mp_pose.PoseLandmark.LEFT_PINKY.value].x , lm[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
        lHip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x , lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        lKnee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x , lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        lAnkle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x , lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        rElbowAng = ang(rShoulder , rElbow , rWrist)
        rWristAng = ang(rElbow , rWrist , rPinky)
        rShoulderAng = ang(rHip , rShoulder , rElbow)
        rHipAng = ang(rKnee , rHip , rShoulder)
        rKneeAng = ang(rAnkle , rKnee , rHip)
        lElbowAng = ang(lShoulder , lElbow , lWrist)
        lWristAng = ang(lElbow , lWrist , lPinky)
        lShoulderAng = ang(lHip , lShoulder , lElbow)
        lHipAng = ang(lKnee , lHip , lShoulder)
        lKneeAng = ang(lAnkle , lKnee , lHip)


        #Rounding Off to 2 decimal places

        rElbowAng = round(rElbowAng , 2)
        rWristAng = round(rWristAng , 2)
        rShoulderAng = round(rShoulderAng , 2)
        rHipAng = round(rHipAng , 2)
        rKneeAng = round(rKneeAng , 2)
        lElbowAng = round(lElbowAng , 2)
        lWristAng = round(lWristAng , 2)
        lShoulderAng = round(lShoulderAng , 2)
        lHipAng = round(lHipAng , 2)
        lKneeAng = round(lKneeAng , 2)


        cv2.putText(img , str(rElbowAng) , tuple(np.multiply(rElbow , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
        cv2.putText(img , str(rWristAng) , tuple(np.multiply(rWrist , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
        cv2.putText(img , str(lElbowAng) , tuple(np.multiply(lElbow , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
        cv2.putText(img , str(lWristAng) , tuple(np.multiply(lWrist , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)

        d = [fNum , rElbowAng , rWristAng , rShoulderAng , rHipAng , rKneeAng , lElbowAng , lWristAng , lShoulderAng , lHipAng , lKneeAng]
        data.append(d)



    except:
        pass

    #Marking Landmarks On the Video
    img.flags.writeable = True
    img = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks , mp_pose.POSE_CONNECTIONS
            )


    out.write(img)

with open(sys.argv[2] , 'w' , encoding='UTF8' , newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data)

poseObj.close()
cap.release()
out.release()

