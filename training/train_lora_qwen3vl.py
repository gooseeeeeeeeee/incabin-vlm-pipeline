"""Qwen3-VL-4B LoRA SFT on flattened (image,q,a) VQA data. transformers 5.13 + peft."""
import json,torch,random,sys
from torch.utils.data import Dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
M='/home/exouser/models/qwen3vl4b'; DATA=sys.argv[1]; OUT=sys.argv[2]
MAXN=int(sys.argv[3]) if len(sys.argv)>3 else 0
proc=AutoProcessor.from_pretrained(M,max_pixels=200704)
model=Qwen3VLForConditionalGeneration.from_pretrained(M,torch_dtype=torch.bfloat16,device_map='cuda')
model.config.use_cache=False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
model=get_peft_model(model,lora); model.print_trainable_parameters()
rows=json.load(open(DATA)); random.seed(0); random.shuffle(rows)
if MAXN: rows=rows[:MAXN]
print('train rows:',len(rows))
class DS(Dataset):
    def __len__(self): return len(rows)
    def __getitem__(self,i): return rows[i]
def collate(batch):
    r=batch[0]
    msgs=[{'role':'user','content':[{'type':'image','image':r['image']},{'type':'text','text':r['q']}]},
          {'role':'assistant','content':[{'type':'text','text':r['a']}]}]
    full=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
    prompt=proc.apply_chat_template(msgs[:1],tokenize=False,add_generation_prompt=True)
    imgs,vids=process_vision_info(msgs)
    enc=proc(text=[full],images=imgs,videos=vids,return_tensors='pt')
    penc=proc(text=[prompt],images=imgs,videos=vids,return_tensors='pt')
    plen=penc['input_ids'].shape[1]
    labels=enc['input_ids'].clone(); labels[:, :plen]=-100
    enc['labels']=labels
    return {k:v for k,v in enc.items()}
args=TrainingArguments(output_dir=OUT,per_device_train_batch_size=1,gradient_accumulation_steps=8,
    num_train_epochs=1,learning_rate=2e-5,bf16=True,logging_steps=10,save_steps=1000,save_total_limit=1,
    warmup_ratio=0.03,lr_scheduler_type='cosine',report_to='none',dataloader_num_workers=2,remove_unused_columns=False)
tr=Trainer(model=model,args=args,train_dataset=DS(),data_collator=collate)
tr.train()
model.save_pretrained(OUT); print('SAVED',OUT)
