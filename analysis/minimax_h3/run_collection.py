import argparse,sys,json,time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--out",default="collection_seed42")
parser.add_argument("--seed",type=int,default=42)
args=parser.parse_args()
sys.argv=[sys.argv[0],"--disable-dynamic-vram","--use-pytorch-cross-attention"]
from analysis.minimax_h3.runtime import *
from analysis.minimax_h3.support import Collection,patch_model
conditioning,meta=prepare_conditioning(["starfire","albedo"])
model=load_model(["starfire","albedo"])
model.model_options["transformer_options"]["freefuse_bypass_manager"].disable_lora()
state=Collection(meta["positions"])
collecting=patch_model(model,collection=state)
def callback(step,x0,x,total):
    state.current_step=step+1
    print("COLLECT_PROGRESS",step+1,len(state.captures),flush=True)
result,settings=sample(collecting,conditioning,seed=args.seed,callback=callback)
target=BASE/"validation"/args.out
save_latents(result,target)
torch.save(state.captures,target/"captures.pt")
summary=[]
for (step,block),capture in sorted(state.captures.items()):
    summary.append({"step":step,"block":block,"grid":capture["grid"],
        "stats":{name:{"min":float(values.min()),"max":float(values.max()),"std":float(values.std())}
                 for name,values in capture["logits"].items()}})
(target/"capture_summary.json").write_text(json.dumps(summary,indent=2))
print("COLLECTION_COMPLETE",len(state.captures),flush=True)
decode_and_save(result,target,{**meta,**settings,"freefuse_phase":"LoRA-disabled region collection",
                              "collection_blocks":list(state.blocks),"collection_steps":list(state.steps)})
