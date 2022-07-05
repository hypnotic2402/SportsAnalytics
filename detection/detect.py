import sys
import cv2
import mediapipe as mp
import numpy as np
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

while cap.isOpened() == True:
    success , img = cap.read()
    if not success:
        break

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

        lShoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x , lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        lElbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x , lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        lWrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x , lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        lPinky = [lm[mp_pose.PoseLandmark.LEFT_PINKY.value].x , lm[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
        
        rElbowAng = ang(rShoulder , rElbow , rWrist)
        rWristAng = ang(rElbow , rWrist , rPinky)
        lElbowAng = ang(lShoulder , lElbow , lWrist)
        lWristAng = ang(lElbow , lWrist , lPinky)


        #Rounding Off to 2 decimal places

        rElbowAng = round(rElbowAng , 2)
        rWristAng = round(rWristAng , 2)
        lElbowAng = round(lElbowAng , 2)
        lWristAng = round(lWristAng , 2)


        cv2.putText(img , str(rElbowAng) , tuple(np.multiply(rElbow , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
        cv2.putText(img , str(rWristAng) , tuple(np.multiply(rWrist , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
        cv2.putText(img , str(lElbowAng) , tuple(np.multiply(lElbow , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
        cv2.putText(img , str(lWristAng) , tuple(np.multiply(lWrist , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)

    except:
        pass

    #Marking Landmarks On the Video
    img.flags.writeable = True
    img = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks , mp_pose.POSE_CONNECTIONS
            )


    out.write(img)

poseObj.close()
cap.release()
out.release()

