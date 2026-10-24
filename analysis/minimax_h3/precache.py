
import sys,json
from pathlib import Path
sys.argv=[sys.argv[0],"--disable-dynamic-vram","--use-pytorch-cross-attention"]
from analysis.minimax_h3.runtime import *
clip,=CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_bf16.safetensors",type="minimax")
for names in [["starfire"],["albedo"],["chel"],["chel","albedo"]]:
 prompt=prompt_for(names)
 target=BASE/"validation"/("conditioning_"+"_".join(names)+".pt")
 positions=concept_positions(clip,prompt,{**{name:CONCEPTS[name] for name in names},"__background__":BACKGROUND})
 with torch.inference_mode():conditioning=clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
 def cpu(value):
  if torch.is_tensor(value):return value.detach().cpu()
  if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
  if isinstance(value,list):return [cpu(v) for v in value]
  return value
 conditioning=cpu(conditioning)
 data={"prompt":prompt,"concepts":{n:CONCEPTS[n] for n in names},"background":BACKGROUND,
       "positions":positions,"text_shape":list(conditioning[0][0].shape)}
 torch.save(conditioning,target);target.with_suffix(".json").write_text(json.dumps(data,indent=2))
 print("CONDITIONING_READY",names,data["text_shape"],flush=True)
