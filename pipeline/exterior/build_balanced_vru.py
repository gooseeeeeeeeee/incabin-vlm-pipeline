import json,re,random
from collections import Counter,defaultdict
random.seed(42)
D=json.load(open("/home/exouser/v21demo/train_flat.json"))
inv=defaultdict(lambda:{"ped":None,"cyc":False,"moto":False,"seen":False})
for x in D:
    q=x["q"].lower(); a=x["a"].lower(); im=x["image"]; inv[im]["seen"]=True
    if q.startswith("how many pedestrians"):
        m=re.search(r"\d+",a)
        if m: inv[im]["ped"]=int(m.group())
    if q.startswith("is the cyclist") and ("bicycle" in a or "cyclist" in a): inv[im]["cyc"]=True
    if q.startswith("is there a motorcyclist") and "motorcyclist" in a and "no motorc" not in a: inv[im]["moto"]=True
yn=[]
for im,v in inv.items():
    if not v["seen"]: continue
    # 只对 vru 图(有 ped 标注的)生成,保证 cyc/moto 的 None=No 推断成立
    if v["ped"] is None: continue
    yn.append({"image":im,"q":"Is there at least one pedestrian in this image? Answer yes or no only.",
               "a":"Yes." if v["ped"]>0 else "No."})
    yn.append({"image":im,"q":"Is there a cyclist (person riding a bicycle) in this image? Answer yes or no only.",
               "a":"Yes." if v["cyc"] else "No."})
    yn.append({"image":im,"q":"Is there a motorcyclist in this image? Answer yes or no only.",
               "a":"Yes." if v["moto"] else "No."})
c=Counter(s["a"] for s in yn)
print("presence QA:",len(yn)," 分布:",dict(c))
byq={}
for s in yn: byq.setdefault(s["q"][:20],Counter())[s["a"]]+=1
for k,v in byq.items(): print("  ",k,dict(v))
# 砍过量模板
kept=[]; seen=Counter(); random.shuffle(D)
for x in D:
    if x["q"].lower().startswith("how many pedestrians"):
        if seen["ped"]<250: kept.append(x); seen["ped"]+=1
    else: kept.append(x)
print("原 %d -> 砍模板 %d"%(len(D),len(kept)))
final=kept+yn*2
random.shuffle(final)
print("最终训练量:",len(final)," presence(x2):",len(yn)*2)
json.dump(final,open("/home/exouser/v21demo/train_v22.json","w"))
