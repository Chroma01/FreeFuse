import argparse,sys,json
parser=argparse.ArgumentParser()
parser.add_argument("--mask",required=True)
parser.add_argument("--mode",choices=["route","bias","combined"],default="combined")
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--out",required=True)
args=parser.parse_args()
sys.argv=[sys.argv[0],"--disable-dynamic-vram","--use-pytorch-cross-attention"]
from analysis.minimax_h3.runtime import *
from analysis.minimax_h3.support import patch_model
conditioning,meta=prepare_conditioning(["starfire","albedo"])
model=load_model(["starfire","albedo"])
if args.mask=="manual":
    left=torch.zeros(37,24,24);left[:,:,:12]=1
    masks={"starfire":left,"albedo":1-left}
    mask_source="manual left/right diagnostic; NOT automatic FreeFuse"
else:
    source=BASE/"validation/masks"/args.mask/"masks.pt"
    masks=torch.load(source,map_location="cpu",weights_only=True)
    mask_source=str(source.relative_to(BASE))
model=patch_model(model,masks=masks,positions=meta["positions"],
    route=args.mode in ("route","combined"),bias=args.mode in ("bias","combined"))
result,settings=sample(model,conditioning,seed=args.seed)
target=BASE/"validation"/args.out
save_latents(result,target)
decode_and_save(result,target,{**meta,**settings,"loras":["starfire","albedo"],
    "freefuse_mode":args.mode,"mask_source":mask_source,"bias_negative":4,"bias_positive":2})

