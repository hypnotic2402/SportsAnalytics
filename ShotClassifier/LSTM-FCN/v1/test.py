import detect as dt
from predict import givePred
import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2
import sys
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def model(file_name, outdir):
    data = np.array(dt.data_csv(file_name))
    data = data[:, 1:11]
    data = data[0:41, :]
    scalar = StandardScaler()
    data = scalar.fit_transform(data)
    print(data.shape)

    modelpath = 'badminton2.h5'
    prediction = givePred(data, modelpath)
    return prediction


def output_vid(out_dir, file_name):
    prediction = model(file_name, out_dir)

    cap = cv2.VideoCapture(file_name)
    poseObj = mp_pose.Pose(min_detection_confidence=0.5 , min_tracking_confidence=0.5)

    if cap.isOpened() == False:
        print("File not opened successfully")
        quit(1)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    outdir, inputflnm = file_name[:file_name.rfind(
        '/')+1], file_name[file_name.rfind('/')+1:]
    outdir = out_dir
    print(outdir, inputflnm)
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
        
            rElbowAng = dt.ang(rShoulder , rElbow , rWrist)
            rWristAng = dt.ang(rElbow , rWrist , rPinky)
            rShoulderAng = dt.ang(rHip , rShoulder , rElbow)
            rHipAng = dt.ang(rKnee , rHip , rShoulder)
            rKneeAng = dt.ang(rAnkle , rKnee , rHip)
            lElbowAng = dt.ang(lShoulder , lElbow , lWrist)
            lWristAng = dt.ang(lElbow , lWrist , lPinky)
            lShoulderAng = dt.ang(lHip , lShoulder , lElbow)
            lHipAng = dt.ang(lKnee , lHip , lShoulder)
            lKneeAng =dt.ang(lAnkle , lKnee , lHip)


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

            mt = (15, 15)
            cv2.putText(img , str(rElbowAng) , tuple(np.multiply(rElbow , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
            cv2.putText(img , str(rWristAng) , tuple(np.multiply(rWrist , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
            cv2.putText(img , str(lElbowAng) , tuple(np.multiply(lElbow , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
            cv2.putText(img , str(lWristAng) , tuple(np.multiply(lWrist , [frame_width , frame_height]).astype(int)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,255,255) , 2 , cv2.LINE_AA)
            cv2.putText(img, 'ShotType: '+ str(prediction), mt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) , 2 , cv2.LINE_AA)

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


# file_name = '../../../test_vids/vid2_s1.mp4'
outdir = '../../../output_vids/'
path = '../../../test_vids'
dir_list = os.listdir(path)
print(dir_list)
for i in dir_list:
    file_name = path + '/' +i
    # print(file_name)
    output_vid(outdir, file_name)
    