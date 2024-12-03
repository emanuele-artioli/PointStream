import math

from torch.distributed import group
from ultralytics import YOLO
import cv2,os

path='/home/farzad/Desktop/onGithub/CGAN/video/'



def find_closest_player(p,p_dic):
    min_dis=10**5
    selected_group=''
    for pl in p_dic.keys():
        d=math.sqrt((float(pl[0])-float(p[0]))**2+(float(pl[1])-float(p[1]))**2)+math.sqrt((float(pl[2])-float(p[2]))**2+(float(pl[3])-float(p[3]))**2)
        if d<min_dis:
            min_dis=d
            selected_group=pl
    return selected_group,min_dis

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # You can use 'yolov8n.pt', 'yolov8s.pt', or larger models

# Load your video frame (replace with the correct path to the frame)
frames=os.listdir(path+'frames/')
frames_sorted = sorted(frames, key=lambda x: int(x.split('_')[1].split('.')[0]))

fr_index=0
players_dic={}
store_img = {}

for fr in frames_sorted:
    frame_path = path+'frames/'+fr
    fr_num=fr.split('_')[1].split('.')[0]
    image = cv2.imread(frame_path)

    # Perform object detection
    results = model(image)

    # Visualize detection results
    # annotated_frame = results[0].plot()  # Draw bounding boxes and labels on the frame

    # Save the result with player bounding boxes
    # cv2.imwrite('player_extracted_yolov8.jpg', annotated_frame)

    # Extract just the players (filter by 'person' class)
    p_num=1

    detected_player=[]
    for detection in results[0].boxes:
        if detection.cls == 0:  # '0' is the class ID for 'person' in COCO dataset
            # Get bounding box coordinates, confidence score, and class label
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            confidence = detection.conf[0]  # Confidence score
            class_id = detection.cls[0]  # Class ID (should be 0 for 'person')
            if class_id==0:
                # Extract player object using the bounding box coordinates
                player = image[y1:y2, x1:x2]
                if (y2-y1)<200 and (x2-x1)<200:
                    continue
                detected_player.append([x1,y1,x2,y2,'fr_' + fr_num + '_p_' + str(p_num) + '.jpg'])

                ## Save each player as a separate file
                # cv2.imwrite(path+'players/fr_'+fr_num+'_p_'+str(p_num)+'.jpg', player)
                store_img['fr_' + fr_num + '_p_' + str(p_num) + '.jpg']=player
                p_num+=1

    if fr_index == 0:
        for p in detected_player:
            players_dic[(p[0],p[1],p[2],p[3])] = [p[4]]
    else:
        for p in detected_player:
            group_p,mindist=find_closest_player(p,players_dic)
            if mindist<150:
                players_dic[group_p].append(p[4])
                players_dic[(p[0],p[1],p[2],p[3])]=players_dic[group_p]
                del(players_dic[group_p])
            else:
                players_dic[(p[0], p[1], p[2], p[3])] = [p[4]]

    fr_index+=1

g_id=0
for p in players_dic.keys():
    g_id += 1
    os.system('mkdir '+path+str(g_id))
    for i in players_dic[p]:
        cv2.imwrite(path+str(g_id)+'/'+i, store_img[i])
