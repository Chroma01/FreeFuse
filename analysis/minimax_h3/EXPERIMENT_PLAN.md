# MiniMax H3 / FreeFuse feasibility experiment

Question: can training-free FreeFuse compose two H3 character/style LoRAs in video,
with automatic regions that remain useful over time?

This is an experiment branch, not a released H3 backend. Use native ComfyUI H3
at a pinned revision and official full BF16 FL2VA weights. Full weights avoid
silently dropping AdaLN tensors trained against the unpruned model. Do not add
turbo, sparse attention or model quantization to the first comparison.

Assumptions recorded before implementation:
- Start with text-to-audio-video, one prompt and one clip. No reference images
  are supplied, so native reference conditioning cannot explain an improvement.
- Video, audio, text and reference rows are separate packed segments. Route only
  the target video rows. Other rows keep the baseline LoRA contribution.
- Spatial routing requires a mask volume in the actual latent (time,height,width)
  patch grid. A still-image mask is not evidence of video support.
- The initial positive-guidance setting is CFG=1, matching the distilled release.
- A manual left/right mask is an explicitly labelled diagnostic upper bound,
  not an automatically detected FreeFuse result.
- Public LoRA training quality may limit identity: generate each adapter alone
  as an independent reference. The Teen Titans adapter is a style/character LoRA,
  not a dedicated Starfire-only adapter.
- The target claim concerns visual composition. Joint audio can change through
  video/audio attention; we do not claim identical or improved audio.
- A tiny / low-resolution run checks execution only. Quality comparisons use
  native 768x768, 124 frames (~5.17 seconds at 24 fps), 20 res_multistep/simple
  steps, video/audio shifts 12/3 and identical seeded joint noise.

Run order:
1. Native tiny-model parity, packed-segment routing and attention-bias tests.
2. Actual native generation with official weights; inspect LoRA targets/coverage.
3. Same-seed global two-LoRA baseline; independent single-LoRA references.
4. LoRA-disabled collection at several layers/steps. Save raw maps and temporal
   mask previews; reject empty regions rather than substituting arbitrary masks.
5. Compare manual-mask routing, automatic routing, bias-only and combined routing
   on the pilot seed. Record parameter choices and failures.
6. If the pilot is promising, confirm on fresh seeds and another composition.

Evidence saved with every run: code/model revisions and weight hashes, exact
prompt and concept spans, seeds, latent geometry, sampling/collection/routing
parameters, LoRA target counts, wall time, raw maps/masks and generated video.
Report pilot evidence as qualitative feasibility, not a general benchmark.

