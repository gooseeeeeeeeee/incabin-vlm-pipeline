import xml.etree.ElementTree as ET, glob, os, json, re, random
from collections import Counter, defaultdict
random.seed(5)
ANN=os.path.expanduser("~/jaad/repo/annotations")
plan=[]  # (video, frame, label)
percv=Counter()
for xml in sorted(glob.glob(ANN+"/*.xml")):
    vid=os.path.basename(xml).replace(".xml","")
    clip=int(re.search(r"video_(\d+)",vid).group(1))
    if clip>=60: continue  # 训练段;held-out 用 60+
    try: root=ET.parse(xml).getroot()
    except: continue
    frame_state=defaultdict(lambda:{"cross":False,"ped":False})
    for track in root.findall("track"):
        if track.get("label") not in ("pedestrian","ped"): continue
        for box in track.findall("box"):
            if box.get("outside")=="1": continue
            f=int(box.get("frame"))
            cr=None
            for at in box.findall("attribute"):
                if at.get("name")=="cross": cr=at.text
            frame_state[f]["ped"]=True
            if cr=="crossing": frame_state[f]["cross"]=True
    yes=[f for f,s in frame_state.items() if s["cross"]]
    no=[f for f,s in frame_state.items() if s["ped"] and not s["cross"]]
    # 每 clip 最多取 5 yes + 5 no,均匀间隔避免近邻帧雷同
    def spaced(lst,k):
        lst=sorted(lst)
        if len(lst)<=k: return lst
        step=len(lst)//k; return [lst[i*step] for i in range(k)]
    for f in spaced(yes,5): plan.append((vid,f,"yes")); percv["yes"]+=1
    for f in spaced(no,5):  plan.append((vid,f,"no"));  percv["no"]+=1
json.dump(plan,open(os.path.expanduser("~/jaad/frame_plan.json"),"w"))
print("采样帧计划:",len(plan),dict(percv))
print("涉及 clips:",len(set(p[0] for p in plan)))
