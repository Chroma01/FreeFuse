import argparse,sys,json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--names",default="starfire,albedo")
parser.add_argument("--width",type=int,default=768)
parser.add_argument("--height",type=int,default=768)
parser.add_argument("--length",type=int,default=124)
parser.add_argument("--steps",type=int,default=20)
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--out",default="native_global")
args=parser.parse_args()
sys.argv=[sys.argv[0],"--disable-dynamic-vram","--use-pytorch-cross-attention"]
from analysis.minimax_h3.runtime import *
names=args.names.split(",")
conditioning,prompt_meta=prepare_conditioning(names)
model=load_model(names)
result,settings=sample(model,conditioning,args.width,args.height,args.length,args.steps,args.seed)
target=BASE/"validation"/args.out
save_latents(result,target)
decode_and_save(result,target,{**prompt_meta,**settings,"loras":names,"freefuse":False})

