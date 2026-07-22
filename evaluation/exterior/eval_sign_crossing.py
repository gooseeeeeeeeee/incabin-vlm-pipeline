import json,os,torch,re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
R=os.path.expanduser("~/v21demo"); H=R+"/heldout"
BASE="/home/exouser/models/qwen3vl4b"; ADAPTER=R+"/q3_v21_lora"
china=json.load(open(H+"/china_test.json")); vru=json.load(open(H+"/vru_test.json"))
def ip(p): return H+"/data/"+p.split("/data/")[-1]
proc=AutoProcessor.from_pretrained(BASE,max_pixels=401408)
model=Qwen3VLForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map="cuda").eval()
def ask(m,img,q,mx=32):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q}]}]
    t=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)
    ins=proc(text=[t],images=[img],return_tensors="pt").to("cuda")
    with torch.no_grad(): o=m.generate(**ins,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
QS="Is the pedestrian in this image about to cross or crossing the road? Answer yes or no only."
QC="What does this traffic sign mean? Answer with a short phrase."
STOP={"the","a","an","of","to","no","for","and","limit","zone","ahead","sign"}
def sign_ok(ans,meaning):
    a=ans.lower(); words=[w for w in re.findall(r"[a-z]+",meaning.lower()) if w not in STOP and len(w)>=3]
    if not words: words=re.findall(r"[a-z]+",meaning.lower())
    return int(any(w in a for w in words))
def yn_ok(ans,gt): a=ans.lower().strip(); return int(("yes" in a[:6])==(gt.lower()=="yes"))
def run(m,tag):
    cs=vs=0; det={"china":[],"vru":[]}
    for x in china:
        ans=ask(m,ip(x["image"]),QC); ok=sign_ok(ans,x["meaning"]); cs+=ok
        det["china"].append({"gt":x["meaning"],"ans":ans,"ok":ok})
    for x in vru:
        ans=ask(m,ip(x["image"]),QS); ok=yn_ok(ans,x["crossing"]); vs+=ok
        det["vru"].append({"gt":x["crossing"],"ans":ans,"ok":ok})
    print(f"[{tag}] china {cs}/40 ({cs/40*100:.1f}%)  vru {vs}/40 ({vs/40*100:.1f}%)  overall {(cs+vs)}/80 ({(cs+vs)/80*100:.1f}%)",flush=True)
    return {"china":cs,"vru":vs,"detail":det}
base_r=run(model,"BASE")
pm=PeftModel.from_pretrained(model,ADAPTER).eval()
v21_r=run(pm,"V21")
json.dump({"base":base_r,"v21":v21_r},open(R+"/eval_heldout_answers.json","w"),indent=1,ensure_ascii=False)
print("DONE")
