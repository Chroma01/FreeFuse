# Krea2 FreeFuse + AutoPrompt + Depth Control (community workflow)

Contributed by **[Rudolf (GPL1967)](https://github.com/GPL1967) & Agent Cow**
(monomatic, Hamburg) in [PR #32](https://github.com/yaoliliu/FreeFuse/pull/32).
Their workflow combines character LoRAs, an optional global style-LoRA block and
depth-guided composition. Thanks for sharing a practical multi-character setup!

![Contributor's original result](../assets/krea2_depth_autoprompt_result.png)

*The contributor's original 1536 × 1024 result mixes photoreal and anime-style
subjects. It uses the original mixed-loading recipe; the maintained template
below has shared sampling settings, four bypass character loaders and optional
styles disabled by default. It is not a promise of pixel-identical reproduction.*

## Load the workflow

Download [krea2_freefuse_depth_autoprompt_api.json](../freefuse_comfyui/workflows/krea2_freefuse_depth_autoprompt_api.json)
and open it through ComfyUI's workflow menu or drag and drop the JSON.

1. Place Krea2 Turbo, its Qwen3-VL 4B text encoder and Qwen Image VAE in the
   usual ComfyUI model folders. The template uses the filenames listed below.
2. Install the three custom-node dependencies below and place the depth Control
   LoRA in `ComfyUI/models/loras/`.
3. Choose four Krea2 character LoRAs in the FreeFuse loaders. The supplied
   character filenames are placeholders; update their matching text blocks for
   your characters. The example styles in the rgthree block are optional and
   disabled; select your own style files before enabling them.
4. For a ready-to-use composition reference, copy
   [the included result PNG](../assets/krea2_depth_autoprompt_result.png) into
   `ComfyUI/input/krea2_depth_autoprompt_result.png`. This reuses the contributor's
   output as a convenient depth reference, **not their unpublished original
   input**. You can instead select any suitable reference in `LoadImage`.
5. Run the workflow. Inspect the generated depth map and automatic region
   previews alongside the final image. Change the two shared sampling inputs to
   adjust seed and total steps for both phases together.

No manual mask edit is required. The mask-editor nodes remain available for
optional adjustments; saved editor references are empty in the template.

## AutoPrompt from shared text blocks

Each of the four character text blocks feeds both `FreeFuseConceptMap` and a
`StringConcatenate` tree. The background block is also shared. The tree joins all
four descriptions, then appends the background and four-character composition
instruction once:

```text
Character 1 + Character 2 ─┐
                          ├─ four descriptions ─┐
Character 3 + Character 4 ─┘                     ├─ final prompt
Background + composition tail ──────────────────┘
```

Editing a character in one place updates its concept and its occurrence in the
main prompt, preserving FreeFuse's verbatim subject-text requirement. Connect
the same blocks when adding more subjects, and update the composition count.

## Depth control before region collection

The control branch uses [comfyui-krea2-controlnet](https://github.com/facok/comfyui-krea2-controlnet)
with the public [Krea2 depth Control LoRA](https://huggingface.co/Patil/Krea-2-depth-controlnet):

```text
LoadImage → DepthAnythingV2 → Krea2ControlImageEncode ─┐
ModelSamplingAuraFlow → Krea2ControlLoRALoader → Krea2ControlApply
                                                   ↓
FreeFusePhase1Sampler → FreeFuseMaskApplicator → KSampler
```

Depth Anything V2 estimates relative depth. The image-encode node matches the
latent size, normalizes the grayscale control and VAE-encodes it. The control
plugin adds depth features to Krea2's input projection and applies the Control
LoRA weights. Strength starts at 1.0.

Both phases receive the depth-controlled model, so automatic regions are
collected while the model is already conditioned on the reference structure.
This can help pose and composition follow the reference. Depth is a soft spatial
constraint; it neither supplies character identities nor guarantees exact
segmentation. Character descriptions and LoRAs still provide appearance.

To disable depth, connect `ModelSamplingAuraFlow` directly to the Phase 1 model
input, bypassing **both** `Krea2ControlLoRALoader` and `Krea2ControlApply`. The
unused reference/preprocessor preview branch can also be removed.

## Models and custom nodes

| Component | Template filename / source |
|---|---|
| Krea2 Turbo | `diffusion_models/krea2_turbo_fp8_scaled.safetensors` |
| Native Krea2 text encoder | `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`, `CLIPLoader type=krea2` |
| Qwen Image VAE | `vae/qwen_image_vae.safetensors` |
| Base model downloads | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) |
| Depth Control LoRA | `loras/depth-control-lora.safetensors`, [Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet) |
| Control nodes | [facok/comfyui-krea2-controlnet](https://github.com/facok/comfyui-krea2-controlnet) |
| `DepthAnythingV2Preprocessor` | [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux), `depth_anything_v2_vitl.pth` |
| Optional global styles | [rgthree-comfy](https://github.com/rgthree/rgthree-comfy), `Power Lora Loader (rgthree)` |

The serialized graph contains the rgthree node even when its style entries are
off, so install rgthree to load it unchanged. To omit that dependency, remove the
style node and connect `UNETLoader` and `CLIPLoader` directly to the first
FreeFuse loader. Install only Krea2-compatible character/style LoRAs.

## Defaults and practical validation

The maintained template uses four bypass character loaders at model strength
0.85, CLIP strength 0, CFG 1, 8 `er_sde`/`simple` steps and one shared seed. It
collects at step 3, block 10. These defaults keep all intended model adapters
active and the two phases on the same sampling configuration. Global styles,
when enabled, affect the whole image.

The original contribution used mixed bypass/simple loaders and different
sampling configurations between the two phases. Maintainer trials in native
ComfyUI successfully ran that recipe with three public character LoRAs and a
three-person composition reference, with depth off/on, plus an all-bypass
variant. Depth guidance brought poses and framing closer to the reference in
that example. These are practical compatibility checks, not a multi-seed quality
benchmark or verification of the original four identities.

Original contribution: Rudolf (idea and direction) · Agent Cow (node wiring).
