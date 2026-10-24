import sys,argparse,json
parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True);a=parser.parse_args()
sys.argv=[sys.argv[0],"--disable-dynamic-vram","--use-pytorch-cross-attention"]
from analysis.minimax_h3.runtime import *
from analysis.minimax_h3.support import make_mask_volumes
torch.set_num_threads(1)
source=BASE/"validation"/f"collection_seed{a.seed}"/"captures.pt"
data=torch.load(source,map_location="cpu",weights_only=True)[(3,20)]
masks=make_mask_volumes(data["logits"],data["grid"],temperature=100000,temporal_window=3)
target=BASE/"validation/masks"/f"seed{a.seed}_s3_b20_t100000";target.mkdir(parents=True,exist_ok=True)
torch.save(masks,target/"masks.pt")
(target/"selection.json").write_text(json.dumps({"pilot_seed":42,"step":3,"block":20,"temperature":100000,"temporal_window":3,"heldout_seed":a.seed},indent=2))
print("FIXED_MASK_READY",target,flush=True)

