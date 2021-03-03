import cv2

#connect camera
cap = cv2.VideoCapture(0)
success, frame1 = cap.read()
success, frame2 = cap.read()

#set up the video recorder
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('detected_object3.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
motion_list = [ None, None ] 

#start recording
while True:
    #get difference between frames to detect the moving object
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh, None, iterations=4)
    
    cont, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #get countours and draw rectangle on the biggest countours in the camera
    for cnt in cont:
    #     param = cv2.arcLength(cnt, True)
    #     aprox = cv2.approxPolyDP(cnt, param, True)
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < 10000:
                continue
        # cv2.drawContours(frame1, cnt, -1, (255, 0, 0), 2)
        
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #start recording the moving object
        writer.write(frame1)
     
    cv2.imshow('1', frame1)
    cv2.imshow('2', blur)
    cv2.imshow('3', thresh)
    cv2.imshow('4', dilate)
    
    frame1 = frame2
    
    success, frame2 = cap.read()
    
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
#release the recordings
writer.release()  
cap.release()
cv2.destroyAllWindows()

''' another way to track moving objects using built in subtractor'''


cap2 = cv2.VideoCapture(0)

fgseg = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    
    _, frame = cap2.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    mask = fgseg.apply(gray)
    
    dilate = cv2.dilate(mask, (3, 3), iterations = 10)
    
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in cnts:
        if cv2.contourArea(cnt) < 4000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    
    # morph = cv2.morphologyEx(frame, cv2.MORPH_OPEN, (5, 5))
    
    cv2.imshow('thresh', dilate)
    cv2.imshow('real', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap2.release()
cv2.destroyAllWindows()























