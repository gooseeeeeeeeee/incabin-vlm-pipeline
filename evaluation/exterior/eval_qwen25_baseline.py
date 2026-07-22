import json,os,torch,re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
R=os.path.expanduser("~/v21demo"); H=R+"/heldout"
BASE="/home/exouser/models/qwen25vl7b"; ADAPTER=R+"/v20_lora"
china=json.load(open(H+"/china_test.json")); vru=json.load(open(H+"/vru_test.json"))
def ip(p): return H+"/data/"+p.split("/data/")[-1]
proc=AutoProcessor.from_pretrained(BASE,max_pixels=401408)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.float16,device_map="cuda").eval()
def ask(m,img,q,mx=32):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q}]}]
    t=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)
    ii,vi=process_vision_info(ms); ins=proc(text=[t],images=ii,videos=vi,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=m.generate(**ins,max_new_tokens=mx,do_sample=False)
    return proc.batch_decode(o[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
QS="Is the pedestrian in this image about to cross or crossing the road? Answer yes or no only."
QC="What does this traffic sign mean? Answer with a short phrase."
STOP={"the","a","an","of","to","no","for","and","limit","zone","ahead","sign"}
def sign_ok(ans,meaning):
    a=ans.lower(); ws=[w for w in re.findall(r"[a-z]+",meaning.lower()) if w not in STOP and len(w)>=3]
    if not ws: ws=re.findall(r"[a-z]+",meaning.lower())
    return int(any(w in a for w in ws))
def yn_ok(ans,gt): a=ans.lower().strip(); return int(("yes" in a[:6])==(gt.lower()=="yes"))
def run(m,tag):
    cs=vs=0
    for x in china: cs+=sign_ok(ask(m,ip(x["image"]),QC),x["meaning"])
    for x in vru: vs+=yn_ok(ask(m,ip(x["image"]),QS),x["crossing"])
    print(f"[{tag}] china {cs}/40 vru {vs}/40 overall {cs+vs}/80",flush=True)
    return {"china":cs,"vru":vs}
r_base=run(model,"Q2.5-BASE")
pm=PeftModel.from_pretrained(model,ADAPTER).eval()
r_v20=run(pm,"V20")
json.dump({"q25_base":r_base,"v20":r_v20},open(R+"/eval_v20_answers.json","w"))
print("V20_EVAL_DONE")
