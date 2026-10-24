# 1.3.0

- Add native MiniMax H3 text-to-audio-video composition through the existing FreeFuse nodes, with temporal mask volumes, packed video-only LoRA routing and memory-efficient video-to-text attention bias.
- Add complete UI/API comparison workflows with shared prompt, seed and schedule, native video/audio decoding and MP4 outputs. Initial scope is T2VA, one video, CFG 1 and one collection block.
- Preserve temporal mask previews and fail clearly for incompatible video grids or unsupported reference/guide conditioning.
- Keep LoRA disable state across rebuilt hooks, and compose standalone H3 attention bias with existing spatial routing.
- Preserve bypass adapters when native fused input-activation helpers are enabled; restore the module's previous flag on ejection.
- Support Unicode concept spans with ComfyUI's slow Qwen tokenizer for MOSS Image.

# 1.2.0

- Add native Anima Base v1.0 composition with T5-aligned collection, float32 similarity maps, spatial LoRA routing and image-to-text attention bias.
- Add an Anima workflow, reproducible 1024 × 1024 comparisons and isolated character references.

- Keep developer API clients out of the Registry package; runtime debug messages use standard logging.
- Remove the legacy adapter precision override and implicit debug tensor dumps; FP8/GGUF adapters retain floating precision.

# 1.1.0

- Add experimental native MOSS Image Base/SFT support, including split and fused QKV checkpoints.
- Route subject LoRAs with spatial masks and apply concept attention bias only to conditional CFG branches.
- Include a 1024 × 1024 MOSS workflow and reproducible comparison settings.
- Correct Registry dependency metadata so requirement-file comments are not treated as package names.
