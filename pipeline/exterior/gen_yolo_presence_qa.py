import glob,os,json,random
random.seed(9)
from ultralytics import YOLO
# 只用训练图(ft_imgs),含 nuScenes 训练帧;绝不碰 held-out sweeps/nuimages
files=glob.glob(os.path.expanduser("~/v21demo/ft_imgs/*.jpg"))+glob.glob(os.path.expanduser("~/v21demo/ft_imgs/*.png"))
random.shuffle(files); files=files[:3500]
yolo=YOLO("yolov8x.pt")
VEH={2,5,7}; PERSON=0; TL=9
qa=[]
from collections import Counter
c=Counter()
for f in files:
    r=yolo(f,verbose=False,conf=0.35)[0]
    cls=[int(x) for x in r.boxes.cls.tolist()]
    veh=sum(1 for x in cls if x in VEH); ped=int(PERSON in cls); tl=int(TL in cls)
    qa.append({"image":f,"q":"Is there at least one pedestrian in this image? Answer yes or no only.","a":"Yes." if ped else "No."})
    qa.append({"image":f,"q":"Is a traffic light visible in this image? Answer yes or no only.","a":"Yes." if tl else "No."})
    qa.append({"image":f,"q":"How many vehicles (cars, buses, trucks) are visible? Answer with a single number only.","a":str(veh)+"."})
    c["ped_yes"]+=ped; c["ped_no"]+=1-ped; c["tl_yes"]+=tl; c["tl_no"]+=1-tl
json.dump(qa,open(os.path.expanduser("~/v21demo/presence_qa.json"),"w"))
print("presence QA:",len(qa),"图:",len(files))
print("平衡度:",dict(c))
