import sys,json
from pathlib import Path
sys.argv=[sys.argv[0],"--disable-dynamic-vram","--use-pytorch-cross-attention"]
from analysis.minimax_h3.runtime import *
from analysis.minimax_h3.support import make_mask_volumes
from PIL import Image,ImageDraw
torch.set_num_threads(1)
captures=torch.load(BASE/"validation/collection_seed42/captures.pt",weights_only=True)
summary=[]
thumbs=[]
for (step,block),data in sorted(captures.items()):
    vals=data["logits"]
    print("RAW",step,block,{n:round(float(v.std()),3) for n,v in vals.items()},flush=True)
    for temp in (100,1000,4000,20000,100000):
        try:masks=make_mask_volumes(vals,data["grid"],temperature=temp)
        except ValueError as e:
            summary.append({"step":step,"block":block,"temperature":temp,"error":str(e)});continue
        target=BASE/"validation/masks"/f"s{step}_b{block}_t{temp}"
        target.mkdir(parents=True,exist_ok=True)
        torch.save(masks,target/"masks.pt")
        stats={n:{"mean":float(m.float().mean()),"frame_min":float(m.float().mean((-2,-1)).min()),"frame_max":float(m.float().mean((-2,-1)).max())} for n,m in masks.items()}
        summary.append({"step":step,"block":block,"temperature":temp,"stats":stats})
        canvas=Image.new("RGB",(3*144,170),"#17191d")
        draw=ImageDraw.Draw(canvas);draw.text((3,3),f"step {step} block {block} T={temp}",fill="white")
        colors={"starfire":(255,95,95),"albedo":(70,225,165),"__background__":(60,90,150)}
        for col,idx in enumerate((0,data["grid"][0]//2,data["grid"][0]-1)):
            pixels=np.zeros((*data["grid"][1:],3),dtype=np.uint8)
            for n,m in masks.items():pixels+=m[idx].numpy().astype(np.uint8)[...,None]*np.array(colors.get(n,(60,90,150)),dtype=np.uint8)
            canvas.paste(Image.fromarray(pixels).resize((144,144),Image.Resampling.NEAREST),(col*144,24))
        canvas.save(target/"preview.png")
        thumbs.append(canvas)
out=BASE/"validation/masks";out.mkdir(exist_ok=True)
(out/"summary.json").write_text(json.dumps(summary,indent=2))
sheet=Image.new("RGB",(4*432,((len(thumbs)+3)//4)*170),"#17191d")
for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%4)*432,(i//4)*170))
sheet.save(out/"contact.png")
print("MASKS_COMPLETE",len(thumbs),flush=True)

