"""Render only derived review media; original generations stay unchanged."""
from pathlib import Path
from fractions import Fraction
import json, av, numpy as np, torch, shutil
from PIL import Image,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parents[3];VAL=BASE/"validation"
OUT=VAL/"review";OUT.mkdir(exist_ok=True)
shutil.copyfile(Path(__file__).with_name("review.html"),OUT/"index.html")
for source in VAL.glob("*/frame_0000.png"):
    Image.open(source).convert("RGB").save(source.with_name("poster.jpg"),quality=88)
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",24)
small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
def frames(name):
    with av.open(str(VAL/name/"video.mp4")) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
def write(path,values):
    with av.open(str(path),"w") as c:
        s=c.add_stream("libx264",rate=24);s.width=values[0].shape[1];s.height=values[0].shape[0];s.pix_fmt="yuv420p";s.options={"crf":"19","preset":"fast"}
        for i,a in enumerate(values):
            f=av.VideoFrame.from_ndarray(a,format="rgb24");f.pts=i;f.time_base=Fraction(1,24)
            for p in s.encode(f):c.mux(p)
        for p in s.encode():c.mux(p)
for prefix,native,auto in [("pilot","native_global","auto_combined"),("heldout","holdout_native_global","holdout_auto_combined")]:
    if not (VAL/auto/"run.json").exists() or not (VAL/native/"run.json").exists():continue
    a,b=frames(native),frames(auto);assert len(a)==len(b)==124
    header=Image.new("RGB",(1536,64),"#12161c");d=ImageDraw.Draw(header)
    seed=json.loads((VAL/native/"run.json").read_text())["seed"]
    d.text((18,8),"WITHOUT FREEFUSE",font=font,fill="#ff9d9d")
    d.text((786,8),"AUTOMATIC FREEFUSE",font=font,fill="#68e3bb")
    d.text((18,39),f"H3 / seed {seed} / identical prompt, LoRAs and sampling settings",font=small,fill="#c4cbd4")
    values=[np.concatenate((np.asarray(header),np.concatenate((x,y),axis=1)),axis=0) for x,y in zip(a,b)]
    write(OUT/(prefix+"_comparison.mp4"),values)
    Image.fromarray(values[62]).save(OUT/(prefix+"_comparison.jpg"),quality=94)
    sheet=Image.new("RGB",(5*384,2*408),"#12161c");d=ImageDraw.Draw(sheet)
    for row,seq in enumerate((a,b)):
        for col,i in enumerate((0,31,62,93,123)):
            sheet.paste(Image.fromarray(seq[i]).resize((384,384)),(col*384,row*408+24))
            d.text((col*384+5,row*408+3),f"{'without' if row==0 else 'automatic'} / frame {i}",font=small,fill="white")
    sheet.save(OUT/(prefix+"_timeline.jpg"),quality=93)
# Mask colors are shown over the LoRA-disabled region-collection video.
masks=torch.load(VAL/"masks/s3_b20_t100000/masks.pt",map_location="cpu",weights_only=True)
source=frames("collection_seed42");colors={"starfire":(255,95,95),"albedo":(70,225,165)}
expanded=[]
for i in range(37):expanded.extend([i]*((1,4,4,4,4)[i%5]))
assert len(expanded)==len(source)
over=[]
for a,i in zip(source,expanded):
    color=np.zeros((24,24,3),np.uint8);coverage=np.zeros((24,24),bool)
    for n,c in colors.items():
        m=masks[n][i].numpy().astype(bool);color[m]=c;coverage|=m
    rgb=np.asarray(Image.fromarray(color).resize((768,768),Image.Resampling.NEAREST))
    cov=np.asarray(Image.fromarray(coverage).resize((768,768),Image.Resampling.NEAREST))
    out=a.copy();out[cov]=(.55*a[cov]+.45*rgb[cov]).astype(np.uint8)
    over.append(out)
write(OUT/"automatic_mask_overlay.mp4",over)
Image.fromarray(over[62]).save(OUT/"automatic_mask_overlay.jpg",quality=92)
names=["native_global","auto_route","auto_bias","auto_combined","manual_combined"]
sheet=Image.new("RGB",(384*5,420),"#12161c");d=ImageDraw.Draw(sheet)
for i,n in enumerate(names):
    if (VAL/n/"frame_0062.png").exists():
        sheet.paste(Image.open(VAL/n/"frame_0062.png").resize((384,384)),(384*i,32))
        d.text((384*i+6,6),n,font=small,fill="white")
sheet.save(OUT/"ablations.jpg",quality=94)
print("REVIEW_MEDIA_COMPLETE",flush=True)
