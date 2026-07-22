import json,os,sys,torch,re,random,glob
random.seed(13)
from ultralytics import YOLO
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
IMGDIR=sys.argv[1]; TAG=sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 150
EXCLUDE=sys.argv[4] if len(sys.argv)>4 else None  # 泄漏排除 basename 文件
BASE="/home/exouser/models/qwen3vl4b"; ADAPTER="/home/exouser/v21demo/q3_v25_lora"
# 1) 采样图 + 泄漏排除
files=sorted(glob.glob(os.path.join(IMGDIR,"**","*.jpg"),recursive=True)+glob.glob(os.path.join(IMGDIR,"**","*.png"),recursive=True))
if EXCLUDE and os.path.exists(EXCLUDE):
    ex=set(open(EXCLUDE).read().split()); files=[f for f in files if os.path.basename(f) not in ex]
random.shuffle(files); files=files[:N]
print(f"[{TAG}] 评测图: {len(files)}",flush=True)
# 2) YOLO GT
yolo=YOLO("yolov8x.pt")
VEH={2,5,7}; PERSON=0; TL=9
gt={}
for f in files:
    r=yolo(f,verbose=False,conf=0.35)[0]
    cls=[int(c) for c in r.boxes.cls.tolist()]
    gt[f]={"veh":sum(1 for c in cls if c in VEH),"ped":int(PERSON in cls),"tl":int(TL in cls)}
del yolo; torch.cuda.empty_cache()
# 3) 模型问答
proc=AutoProcessor.from_pretrained(BASE,max_pixels=401408)
model=Qwen3VLForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map="cuda").eval()
def ask(m,img,q,mx=20):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q}]}]
    t=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)
    ins=proc(text=[t],images=[img],return_tensors="pt").to("cuda")
    with torch.no_grad(): o=m.generate(**ins,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
QV="How many vehicles (cars, buses, trucks) are visible? Answer with a single number only."
QP="Is there at least one pedestrian in this image? Answer yes or no only."
QT="Is a traffic light visible in this image? Answer yes or no only."
def num(a):
    m=re.search(r"\d+",a); return int(m.group()) if m else -99
def yn(a): return 1 if "yes" in a.lower()[:6] else 0
def run(m):
    v=p=t=0
    for f in files:
        g=gt[f]
        v+=int(abs(num(ask(m,f,QV))-g["veh"])<=1)   # 计数±1容差
        p+=int(yn(ask(m,f,QP))==g["ped"])
        t+=int(yn(ask(m,f,QT))==g["tl"])
    n=len(files)
    return {"veh":round(100*v/n,1),"ped":round(100*p/n,1),"tl":round(100*t/n,1),"overall":round(100*(v+p+t)/(3*n),1)}
rb=run(model)
pm=PeftModel.from_pretrained(model,ADAPTER).eval()
rv=run(pm)
out={"tag":TAG,"n":len(files),"base":rb,"v25":rv}
json.dump(out,open(f"/home/exouser/nusc_eval/obj_{TAG}.json","w"),indent=1)
print(f"[{TAG}] base {rb}")
print(f"[{TAG}] v25  {rv}")
print(f"[{TAG}] 提升 overall: {rv['overall']-rb['overall']:+.1f}  veh {rv['veh']-rb['veh']:+.1f} ped {rv['ped']-rb['ped']:+.1f} tl {rv['tl']-rb['tl']:+.1f}")
print(f"OBJDONE_{TAG}")
