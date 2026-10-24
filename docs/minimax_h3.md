# MiniMax H3 in ComfyUI

FreeFuse supports native MiniMax H3 text-to-audio-video generation with multiple
character/style LoRAs. It collects automatic **time × height × width** regions,
routes each LoRA inside the target video segment, and optionally adds video-to-text
attention bias. The joint audio latent stays in the native sampling pipeline.

Load [the complete workflow](../freefuse_comfyui/workflows/minimax_h3_freefuse_complete.json)
in ComfyUI. It produces two MP4s, with and without FreeFuse, plus a temporal mask
preview. Both branches share the prompt, seed and total step count. The
[API workflow](../freefuse_comfyui/workflows/minimax_h3_freefuse_api.json) can also be
submitted to ComfyUI's `/prompt` endpoint.

## Models

Use a current ComfyUI build with `EmptyMiniMaxH3LatentAV`, native MiniMax text
encoding and H3 block replacement hooks. Validation used ComfyUI commit
`7cebca52227c63b4180ceed448cbd7ad22903b6d` and a single NVIDIA H200.

Download the following from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main):

| ComfyUI directory | Filename |
|---|---|
| `models/diffusion_models/` | `minimax_h3_fl2va_bf16.safetensors` |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_bf16.safetensors` |
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors` |
| `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` |

These full-precision files total about 124 GB on disk. Use `CLIPLoader` type
**minimax** and retain native conditioning metadata. The example uses full BF16
weights; pruned, quantized and sparse/turbo variants have not been validated.
Validation used `--use-pytorch-cross-attention --disable-dynamic-vram`. Start
with PyTorch attention: some alternative kernels do not accept the additional
Q/K channels used by H3 FreeFuse's attention bias.

The example expects these two H3 LoRAs in `models/loras/`:

| Adapter | Filename | Download |
|---|---|---|
| `starfire` | `TEENTITANS_H3_V1C-step00001500.safetensors` | [Teen Titans, version 3367577](https://civitai.com/models/2971649?modelVersionId=3367577) |
| `albedo` | `AlbedoOverlordEng_H3_v1.safetensors` | [Albedo, version 3343083](https://civitai.com/models/2809374?modelVersionId=3343083) |

The Teen Titans adapter covers characters and style; it is not a dedicated
Starfire-only LoRA. Civitai may require login to download. Model weights and
credentials are not included in the workflow. Replace the LoRA filenames,
adapter names, concept descriptions and shared prompt together for other cases.
Each subject description must occur verbatim in the shared prompt.

## Settings

| Setting | Example |
|---|---|
| Output | 768 × 768, 124 frames, 24 fps |
| Latent | `EmptyMiniMaxH3LatentAV`, one video with its audio stream |
| Sampler | `res_multistep`, `simple`, 20 steps, CFG 1 |
| LoRA strengths | Model 1, text encoder 0 |
| Phase 1 | LoRAs disabled, collect step **4**, block **20**, end block **20** |
| Similarity | Temperature 100000 (`0` selects this H3 default), top-k 0.1 |
| Masks | Morphological cleaning on, background scale 0.95, balance iterations 15 |
| Temporal smoothing | `h3_temporal_window=3` latent time positions |
| Attention bias | Negative 4, positive 2, all blocks |

The node's collection step is **one-based**; step 4 is denoising index 3 in the
experiment records. Block indices are zero-based. Phase 1 follows the full
20-step sigma schedule and stops as soon as the selected layer is captured.
Phase 2 restarts from the original joint latent and the same seed.

Use the existing `FreeFuseLoRALoader`, `FreeFuseConceptMap`,
`FreeFuseTokenPositions`, `FreeFusePhase1Sampler`, `FreeFuseMaskApplicator`,
`KSampler` and `FreeFuseMaskPreview` nodes. H3 is detected automatically.
`FreeFuseAttentionBias` can also be used separately after spatial routing.

Masks use the actual packed video grid, **37 × 24 × 24** for this example.
The preview batch contains 37 latent time positions, not 124 decoded video frames.
Do not interpret a still-image editor mask or the audio rows as video positions.
Mask volumes from another video geometry are rejected.

For H3, `enable_token_masking` and `bidirectional` do not add text/audio routing:
LoRA masks and attention bias act on target video rows. The optional attention
bias supports `all`, `last_half` and `none`; the legacy `double_stream_only`
selection is accepted as `all` for H3. Use a single collection block in this release.
Spatial routing covers the main DiT attention and MLP adapters. LoRA targets in
shared conditioning, AdaLN or token-refiner layers remain global.

## Validated scope

Validation covers two seeds of the supplied static-camera, gentle-wave example,
including independent single-LoRA references and with/without comparisons.
The baseline can mix Albedo's horns, eyes and clothing into Starfire; automatic
FreeFuse reduces these visible conflicts in this case. Composition can also
change. This is qualitative evidence for these examples, not a general video
quality or identity benchmark.

This release supports **text-to-audio-video, batch size 1 and CFG 1**. Reference
images, guide/keyframe conditioning, complex motion, crossing subjects and camera
cuts are not validated. The joint audio output may change indirectly through
video/audio attention; audio quality and invariance are not claimed.

If regions are empty or misassigned, inspect the temporal mask preview and try
another collection block/step or similarity temperature. The node reports empty
automatic subject regions as an error rather than silently substituting masks.
