"""Merge sharegpt files, flatten to single-turn (image,q,a), remap image paths by basename to local IMG_ROOT."""
import json,glob,os,sys
IMG_ROOT=sys.argv[1]; OUT=sys.argv[2]; SRCS=sys.argv[3:]
# index all images by basename
idx={}
for f in glob.glob(IMG_ROOT+'/**/*.jpg',recursive=True)+glob.glob(IMG_ROOT+'/**/*.png',recursive=True):
    idx[os.path.basename(f)]=f
print('indexed images:',len(idx))
out=[]; miss=0; miss_ex=[]
for src in SRCS:
    data=json.load(open(src))
    for s in data:
        imgs=s.get('images',[])
        if not imgs: continue
        bn=os.path.basename(imgs[0]); lp=idx.get(bn)
        if not lp:
            miss+=1
            if len(miss_ex)<3: miss_ex.append(bn)
            continue
        conv=s['conversations']; 
        # flatten human/gpt pairs
        for i in range(0,len(conv)-1,2):
            if conv[i]['from']=='human' and conv[i+1]['from']=='gpt':
                q=conv[i]['value'].replace('<image>','').strip()
                a=conv[i+1]['value'].strip()
                if q and a: out.append({'image':lp,'q':q,'a':a})
json.dump(out,open(OUT,'w'),ensure_ascii=False)
print('flattened samples:',len(out),'| missing-image samples:',miss,'ex:',miss_ex)
