import cv2,json,os,random
from collections import defaultdict,Counter
random.seed(5)
plan=json.load(open(os.path.expanduser("~/jaad/frame_plan.json")))
VID=os.path.expanduser("~/jaad/JAAD_clips"); OUT=os.path.expanduser("~/jaad/frames"); os.makedirs(OUT,exist_ok=True)
byv=defaultdict(list)
for vid,f,lab in plan: byv[vid].append((f,lab))
rows=[]; got=Counter()
for vid,fl in byv.items():
    cap=cv2.VideoCapture(f"{VID}/{vid}.mp4")
    if not cap.isOpened(): continue
    for f,lab in fl:
        cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        ok,img=cap.read()
        if not ok: continue
        p=f"{OUT}/{vid}_{f}.jpg"; cv2.imwrite(p,img)
        rows.append({"image":p,"crossing":lab}); got[lab]+=1
    cap.release()
json.dump(rows,open(os.path.expanduser("~/jaad/crossing_frames.json"),"w"))
print("抽出帧:",len(rows),dict(got))
