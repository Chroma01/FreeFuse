"""Isolated native ComfyUI runner utilities for H3 feasibility experiments."""
from pathlib import Path
import sys, json, logging, gc, time, hashlib, subprocess
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent
sys.path[:0]=[str(BASE/"ComfyUI"),str(ROOT)]
import torch, numpy as np, av
import folder_paths
import comfy.sample, comfy.model_management, comfy.utils
from nodes import UNETLoader,CLIPLoader,VAELoader
from comfy_extras.nodes_minimax_h3 import _empty_av_latent
from comfy_extras.nodes_audio import vae_decode_audio
from freefuse_comfyui.freefuse_core.bypass_lora_loader import load_bypass_lora_for_models_fixed
from analysis.minimax_h3.support import concept_positions

logging.basicConfig(level=logging.INFO)
torch.set_grad_enabled(False)
for category in ("diffusion_models","text_encoders","vae","loras"):
    folder_paths.add_model_folder_path(category,str(BASE/"models"/category))

CONCEPTS={
 "starfire":"char_starfire, Starfire, an adult woman with orange skin, long straight red hair and bright green eyes, wearing her purple costume with a silver gorget and a green gemstone, drawn in Teen Titans (2003) style, 2d animation",
 "albedo":"lbd0c1tr0n, Albedo, a demon woman with yellow eyes, slit pupils, long black hair and white horns, wearing a white dress, gold ornaments and white gloves, with black feathered wings, drawn in Japanese anime style",
 "chel":"ChelR, Chel, an adult woman with tan skin, long straight black hair with blunt bangs and brown eyes, wearing golden earrings, a pink-and-white top and a white wrap skirt, drawn in classic 2d animation style",
}
BACKGROUND="a sunny garden with flowers"

def prompt_for(names):
    if len(names)==1:
        shot=f"A single continuous locked-off medium shot of {CONCEPTS[names[0]]}, standing in {BACKGROUND}. She smiles at the camera and slowly raises one hand in a small greeting wave."
    else:
        shot=(f"A single continuous locked-off medium shot of two fully dressed adult women, framed from the waist up, standing shoulder to shoulder in {BACKGROUND}. "
              f"On the left is {CONCEPTS[names[0]]}. On the right is {CONCEPTS[names[1]]}. "
              "They smile at the camera and each slowly raises one hand in a small greeting wave. Their bodies remain separated and they remain on their own sides of the frame for the whole shot.")
    return "integrated_multimodal_description: "+shot+" No camera cuts.\noverall_soundscape: Gentle bird song and light outdoor ambience. No speech.\nnon_diegetic_music: None."


def prepare_conditioning(names):
    cache=BASE/"validation"/("conditioning_"+"_".join(names)+".pt")
    meta=cache.with_suffix(".json")
    prompt=prompt_for(names)
    if cache.exists() and meta.exists():
        data=json.loads(meta.read_text())
        if data["prompt"]!=prompt:raise ValueError("Cached prompt differs; use a new experiment directory.")
        return torch.load(cache,map_location="cpu",weights_only=True),data
    clip,=CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_bf16.safetensors",type="minimax")
    positions=concept_positions(clip,prompt,{**{name:CONCEPTS[name] for name in names},"__background__":BACKGROUND})
    with torch.inference_mode():
        conditioning=clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    def cpu(value):
        if torch.is_tensor(value):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    conditioning=cpu(conditioning)
    data={"prompt":prompt,"concepts":{n:CONCEPTS[n] for n in names},"background":BACKGROUND,
          "positions":positions,"text_shape":list(conditioning[0][0].shape)}
    torch.save(conditioning,cache);meta.write_text(json.dumps(data,indent=2))
    comfy.model_management.unload_all_models()
    del clip
    gc.collect()
    print("CONDITIONING_READY",names,data["text_shape"],flush=True)
    return conditioning,data


def load_model(names):
    model,=UNETLoader().load_unet("minimax_h3_fl2va_bf16.safetensors","default")
    manifest=json.loads((BASE/"reference/lora_manifest.json").read_text())
    expected=0
    for name in names:
        info=next(f for f in manifest if f["adapter"]==name)
        path=BASE/"models/loras"/info["name"]
        weights=comfy.utils.load_torch_file(str(path),safe_load=True)
        model,_=load_bypass_lora_for_models_fixed(model,None,weights,1.0,0.0,adapter_name=name)
        manager=model.model_options["transformer_options"].get("freefuse_bypass_manager")
        count=manager.get_total_adapter_count() if manager is not None else 0
        print("LORA_TARGETS",name,count-expected,"total",count,flush=True)
        if count<=expected:raise ValueError(f"No native H3 targets matched for {name}")
        expected=count
        del weights
    return model


def sample(model,conditioning,width=768,height=768,length=124,steps=20,seed=42,callback=None):
    latent,frame_count=_empty_av_latent(width,height,length)
    noise=comfy.sample.prepare_noise(latent["samples"],seed)
    started=time.monotonic()
    with torch.inference_mode():
        result=comfy.sample.sample(model,noise,steps,1.0,"res_multistep","simple",
            conditioning,[],latent["samples"],seed=seed,callback=callback,
            force_full_denoise=True)
    print("SAMPLING_COMPLETE",time.monotonic()-started,flush=True)
    return result,{"width":width,"height":height,"frames":frame_count,"steps":steps,
        "seed":seed,"cfg":1.0,"sampler":"res_multistep","scheduler":"simple",
        "latent_shapes":[list(t.shape) for t in latent["samples"].unbind()],
        "denoise_seconds":time.monotonic()-started}


def save_video(path,frames,audio=None,sample_rate=32000):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays=(frames.clamp(0,1).mul(255).to(torch.uint8).cpu().numpy()
            if torch.is_tensor(frames) else frames)
    with av.open(str(path),"w") as container:
        stream=container.add_stream("libx264",rate=24)
        stream.width,stream.height=arrays.shape[2],arrays.shape[1]
        stream.pix_fmt="yuv420p";stream.options={"crf":"19","preset":"fast"}
        astream=None
        if audio is not None:
            astream=container.add_stream("aac",rate=sample_rate)
            astream.layout="stereo"
        for index,values in enumerate(arrays):
            frame=av.VideoFrame.from_ndarray(values,format="rgb24")
            frame.pts=index;frame.time_base=Fraction(1,24)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
        if astream is not None:
            wave=audio.detach().float().cpu().numpy() if torch.is_tensor(audio) else audio
            frame=av.AudioFrame.from_ndarray(np.ascontiguousarray(wave),format="fltp",layout="stereo")
            frame.sample_rate=sample_rate;frame.pts=0;frame.time_base=Fraction(1,sample_rate)
            for packet in astream.encode(frame):container.mux(packet)
            for packet in astream.encode():container.mux(packet)
    print("VIDEO_SAVED",path,len(arrays),flush=True)


def decode_and_save(result,target,metadata):
    video,audio=result.unbind()
    vae,=VAELoader().load_vae("minimax_h3_video_vae_fp16.safetensors")
    with torch.inference_mode():
        frames=vae.decode(video)
    if frames.ndim==5:frames=frames.flatten(0,1)
    frames=frames.detach().cpu()
    if len(frames)!=metadata["frames"]:raise ValueError(f"Decoded {len(frames)} frames, expected {metadata['frames']}")
    audio_vae,=VAELoader().load_vae("minimax_h3_audio_vae_fp32.safetensors")
    with torch.inference_mode():track=vae_decode_audio(audio_vae,{"samples":audio})
    save_video(Path(target)/"video.mp4",frames,track["waveform"][0],track["sample_rate"])
    from PIL import Image
    for index in (0,len(frames)//2,len(frames)-1):
        Image.fromarray((frames[index].clamp(0,1).numpy()*255).astype(np.uint8)).save(Path(target)/f"frame_{index:04d}.png")
    Path(target,"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return frames


def save_latents(result,target):
    Path(target).mkdir(parents=True,exist_ok=True)
    torch.save({"video":result.unbind()[0].cpu(),"audio":result.unbind()[1].cpu()},Path(target)/"latents.pt")

