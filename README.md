# ComfyUI-FreeFuse

FreeFuse for ComfyUI: multi-concept LoRA composition with spatial awareness.

## Workflows (Complete only)

- [workflows/flux_freefuse_complete.json](workflows/flux_freefuse_complete.json)
- [workflows/flux2_klein_4b_freefuse_complete.json](workflows/flux2_klein_4b_freefuse_complete.json)
- [workflows/flux2_klein_9b_freefuse_complete.json](workflows/flux2_klein_9b_freefuse_complete.json)
- [workflows/sdxl_freefuse_complete.json](workflows/sdxl_freefuse_complete.json)
- [workflows/zimage_freefuse_complete.json](workflows/zimage_freefuse_complete.json)
- [workflows/krea2_freefuse_with_editor.json](workflows/krea2_freefuse_with_editor.json)

## Installation

```bash
git clone <this-repo>
ln -s /path/to/FreeFuse/comfyui ComfyUI/custom_nodes
```

## Example LoRAs and Prompt (from test_parameters.py)

**LoRA download links**

- Daiyu: https://huggingface.co/lsmpp/freefuse_community_loras/resolve/main/daiyu_lin.safetensors?download=true
- Harry: https://huggingface.co/lsmpp/freefuse_community_loras/resolve/main/harry_potter.safetensors?download=true
- Jinx (Z-Image-Turbo): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/Jinx_Arcane_zit.safetensors?download=true
- Skeletor (Z-Image-Turbo): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/skeletor_zit.safetensors?download=true

> The workflows expect these filenames by default:
> - Flux: harry_potter_flux.safetensors, daiyu_lin_flux.safetensors
> - Flux2.Klein 4B: flux-2-klein-4b.safetensors + qwen_3_4b.safetensors + flux2-vae.safetensors
> - Flux2.Klein 9B: flux-2-klein-9b-fp8.safetensors + qwen_3_8b_fp8mixed.safetensors + flux2-vae.safetensors
> - SDXL: harry_potter_xl.safetensors, daiyu_lin_xl.safetensors
> - Z-Image-Turbo: Jinx_Arcane_zit.safetensors, skeletor_zit.safetensors
> - Krea2 Turbo: krea2_turbo_fp8_scaled.safetensors + qwen3vl_4b_fp8_scaled.safetensors + qwen_image_vae.safetensors
> - Krea2 example LoRAs: FreeFuse/Krea2/Krea 2 - Kim Possible.safetensors, FreeFuse/Krea2/Krea 2 - Violet Parr.safetensors
> If you use the downloads above, rename the files or update the workflow nodes.

**Prompt**

Realistic photography, harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes hugging daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression, autumn leaves blurred in the background, high quality, detailed

**Negative Prompt (SDXL only)**

low quality, blurry, deformed, ugly, bad anatomy

**Concept Map**

- harry: harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes
- daiyu: daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression
- background_text: autumn leaves blurred in the background

## Important Prompt Rule

- Every **subject** `concept_text` (adapter trigger phrase) must appear verbatim in the **main prompt**.
- If any subject concept is missing from the main prompt, `FreeFuseTokenPositions` / `FreeFuseConceptMapSimple` now raises an error in ComfyUI.
- `background_text` is optional for runtime safety: if provided but not found in the main prompt, FreeFuse only prints a warning and continues.

Example:
- `concept_text = "harry potter"` means your main prompt must contain `"harry potter"`.

## Hyperparameters

### Phase 1 (FreeFuse Phase1 Sampler)

- `steps`: Total steps for Phase 2 (keep consistent for the same noise schedule)
- `collect_step`: Which step to collect attention and early-stop
- `collect_block`: Transformer block/layer to extract attention (Flux: `transformer_blocks.<idx>`, Flux2: `single_transformer_blocks.<idx>`, Z-Image: `layers.<idx>`, Krea2: `blocks.<idx>`, SDXL ignored)
- `collect_block_end`: Optional inclusive end index for range-mode collection (Flux/Flux2/Z-Image/Krea2). Set `collect_block_end > collect_block` to enable majority-vote aggregation across blocks.
- `temperature`: Softmax temperature for similarity; 0 = auto (Flux/Flux2=4000, SDXL=300)
- `top_k_ratio`: Ratio of top-k tokens used for similarity
- `disable_lora_phase1`: Disable LoRA in Phase 1 (recommended for cleaner attention)
- `bg_scale`: Background similarity scale (higher = more background)
- `use_morphological_cleaning`: Apply morphological cleanup
- `balance_iterations`: Iterations for balanced argmax (higher = more stable, slower)

### Phase 2 (FreeFuse Mask Applicator)

- `enable_token_masking`: Token-level masking (zero out other concept tokens)
- `enable_attention_bias`: Enable attention bias
- `bias_scale`: Negative bias strength (suppresses wrong concepts)
- `positive_bias_scale`: Positive bias strength (enhances correct concepts)
- `bidirectional`: Flux/Flux2 bidirectional bias (text↔image)
- `use_positive_bias`: Enable positive bias
- `bias_blocks`: Which blocks to apply bias (recommended all or double_stream_only)

### Sampling (KSampler / FluxGuidance)

- Flux uses FluxGuidance for CFG; set KSampler CFG to 1.0
- Flux2.Klein uses CLIPTextEncode + CLIPLoader(type=`flux2`); keep KSampler CFG at 1.0 as a safe default
- Krea2 uses CLIPTextEncode + CLIPLoader(type=`krea2`); the included comparison runner follows the official Krea2 Turbo 8-step KSampler setup
- SDXL uses KSampler CFG directly (recommended 7.0)

## Krea2 CUDA Comparison Test

Krea2 fp8 weights should be tested on a CUDA host. On macOS, ComfyUI auto device selection uses Apple MPS, which cannot run the fp8 Krea2 tensors; use `--device cpu` only for a tiny local compatibility smoke test.

Prepare ComfyUI, the venv, and the official Krea2 model files under `/tmp`:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
LORA_SOURCE_DIR=/tmp/freefuse-krea2/input_loras \
bash freefuse_comfyui/scripts/setup_krea2_cuda_test.sh
```

The setup script expects the two character LoRAs in `LORA_SOURCE_DIR` unless `KIM_LORA_FILE` and `VIOLET_LORA_FILE` are set explicitly.

To stage the local character LoRAs before copying them to a CUDA host:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
bash freefuse_comfyui/scripts/pack_krea2_loras.sh
```

Run the 8-GPU seed matrix after setup:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
OUTPUT_DIR=/tmp/freefuse-krea2/results/krea2_matrix \
bash freefuse_comfyui/scripts/run_krea2_cuda_matrix.sh
```

Or run the full remote flow from the LoRA tar in one command:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
bash freefuse_comfyui/scripts/run_krea2_remote_full_test.sh
```

After choosing the best `krea2_baseline_vs_freefuse.png`, copy it back and update the top-level README:

```bash
python3 freefuse_comfyui/scripts/update_krea2_readme_result.py \
  --contact-sheet /path/to/krea2_baseline_vs_freefuse.png
```

If you copied back the full remote result bundle, import it and then select the best seed explicitly:

```bash
RESULT_BUNDLE=/tmp/freefuse-krea2/krea2_cuda_results.tar.gz \
UPDATE_README=0 \
bash freefuse_comfyui/scripts/import_krea2_cuda_results.sh
```

## Preview Image

The workflows include a preview image:
freefuse_flux_square_1024_output.png. It shows up in the Preview when the workflow loads.

## License

Apache 2.0
