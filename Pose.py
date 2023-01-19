import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


Video = cv2.VideoCapture(0)
while True:
    check, img = Video.read()
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(img)
    mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Camera", img)
    cv2.waitKey(1)


