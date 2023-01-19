import cv2
import mediapipe as mp



Video = cv2.VideoCapture(0)

hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = Video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hands.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    
    h, w, _ = img.shape
    pontos = []
    
    if handsPoints:
        for points in handsPoints:
            #print(points)
            mpDraw.draw_landmarks(img, points, hands.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x*w), int(cord.y*h)
                #cv2.putText(img, str(id), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2)
                pontos.append((cx,cy))
                #print(pontos)
    
    
        dedos = [8, 12, 16, 20]
        contador = 0
        if points:
            if pontos[4][0] > pontos [2][0]:
                contador +=1
            for x in dedos:
                if pontos[x][1] < pontos[x-2][1]:
                    contador +=1

        print(contador)
        cv2.rectangle(img,(80,10), (100,200),(0,0,0),-1)
        cv2.putText(img, str(contador),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,0,0), 5)


    cv2.imshow("Camera", img)
    cv2.waitKey(1)

