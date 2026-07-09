#!/usr/bin/env python3
"""Krea2 checks that do not require torch, ComfyUI, or model weights."""

import importlib.util
import json
from pathlib import Path
import unittest


THIS_FILE = Path(__file__).resolve()
PLUGIN_ROOT_CANDIDATES = (
    THIS_FILE.parents[1],
    THIS_FILE.parents[2] / "freefuse_comfyui",
)
for candidate in PLUGIN_ROOT_CANDIDATES:
    if (candidate / "freefuse_core" / "token_utils.py").exists():
        PLUGIN_ROOT = candidate
        break
else:
    raise RuntimeError("Could not resolve FreeFuse plugin root for Krea2 tests")

TOKEN_UTILS_PATH = PLUGIN_ROOT / "freefuse_core" / "token_utils.py"
WORKFLOW_PATH = PLUGIN_ROOT / "workflows" / "krea2_freefuse_with_editor.json"


def _load_token_utils():
    spec = importlib.util.spec_from_file_location("freefuse_token_utils", TOKEN_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


token_utils = _load_token_utils()


class FakeQwen3VLTokenizer:
    def __init__(self, token_texts):
        self.token_texts = dict(token_texts)

    def decode(self, token_ids):
        return "".join(self.token_texts[int(token_id)] for token_id in token_ids)


class FakeTokenizerBranch:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class FakeTokenizerWrapper:
    def __init__(self, tokenizer):
        self.qwen3vl_4b = FakeTokenizerBranch(tokenizer)


class FakeClip:
    def __init__(self, token_ids, tokenizer):
        self.token_ids = list(token_ids)
        self.tokenizer = FakeTokenizerWrapper(tokenizer)
        self.last_tokenized_text = None

    def tokenize(self, text):
        self.last_tokenized_text = text
        return {"qwen3vl_4b": [[(token_id, 1.0) for token_id in self.token_ids]]}


class Krea2TokenUtilsTests(unittest.TestCase):
    def test_krea2_positions_are_stripped_to_user_prompt_sequence(self):
        tokenizer = FakeQwen3VLTokenizer({
            10: "kimpossible",
            11: " and ",
            12: "violetparr",
        })
        token_ids = [
            token_utils.KREA2_IM_START_ID,
            9000,
            token_utils.KREA2_IM_START_ID,
            token_utils.KREA2_USER_TOKEN_ID,
            token_utils.KREA2_NEWLINE_TOKEN_ID,
            10,
            11,
            12,
        ]
        clip = FakeClip(token_ids, tokenizer)

        result = token_utils.find_concept_positions(
            clip=clip,
            prompts="kimpossible and violetparr",
            concepts={"kimpossible": "kimpossible", "violetparr": "violetparr"},
            model_type="krea2",
        )

        self.assertEqual(result["kimpossible"], [[0]])
        self.assertEqual(result["violetparr"], [[2]])
        self.assertIn("Describe the image by detailing", clip.last_tokenized_text)
        self.assertIn("kimpossible and violetparr", clip.last_tokenized_text)

    def test_krea2_uses_qwen3vl_branch_from_chunked_token_pairs(self):
        token_pairs = {
            "t5xxl": [[(1, 1.0)]],
            "qwen3vl_4b": [[(10, 1.0), (11, 1.0)]],
        }

        self.assertEqual(token_utils._flatten_chunked_token_ids(token_pairs), [10, 11])

    def test_krea2_does_not_direct_tokenizer_fallback_when_clip_tokenize_fails(self):
        tokenizer = FakeQwen3VLTokenizer({10: "kimpossible"})
        clip = FakeClip([10], tokenizer)

        def broken_tokenize(text):
            raise RuntimeError("intentional tokenize failure")

        clip.tokenize = broken_tokenize
        with self.assertRaisesRegex(RuntimeError, "intentional tokenize failure"):
            token_utils.find_concept_positions(
                clip=clip,
                prompts="kimpossible",
                concepts={"kimpossible": "kimpossible"},
                model_type="krea2",
            )

    def test_krea2_model_detection(self):
        class DiffusionModel:
            txtfusion = object()
            blocks = []

        class InnerModel:
            diffusion_model = DiffusionModel()

        class ModelPatcher:
            model = InnerModel()

        self.assertEqual(token_utils.detect_model_type_from_model(ModelPatcher()), "krea2")
        self.assertEqual(token_utils.detect_model_type(model_type_hint="Krea 2"), "krea2")
        self.assertEqual(token_utils.detect_model_type(model_type_hint="k2"), "krea2")


class Krea2WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
            cls.workflow = json.load(handle)
        cls.nodes = {node["id"]: node for node in cls.workflow["nodes"]}

    def test_prompt_concepts_and_lora_names_are_consistent(self):
        prompt = self.nodes[9]["widgets_values"][0]
        token_prompt = self.nodes[8]["widgets_values"][0]
        concept_values = self.nodes[6]["widgets_values"]

        self.assertEqual(token_prompt, prompt)
        self.assertIn("archer-queen", prompt)
        self.assertIn("satoru gojo", prompt)
        self.assertEqual(concept_values[0], "archer-queen")
        self.assertIn("archer-queen, Archer Queen", concept_values[1])
        self.assertEqual(concept_values[2], "satoru-gojo")
        self.assertIn("satoru gojo, white hair, black blindfold", concept_values[3])

        stale_text = "\n".join([prompt] + [str(v) for v in concept_values[:4]])
        for stale in ("Kim Possible", "kimpossible", "Violet Parr", "violetparr", "Jinx", "Skeletor", "Durian"):
            self.assertNotIn(stale, stale_text)

    def test_model_and_lora_paths_are_krea2_specific(self):
        self.assertEqual(self.nodes[109]["widgets_values"][:2], ["qwen3vl_4b_fp8_scaled.safetensors", "krea2"])
        self.assertEqual(self.nodes[108]["widgets_values"][0], "krea2_turbo_fp8_scaled.safetensors")
        self.assertEqual(self.nodes[3]["widgets_values"][0], "qwen_image_vae.safetensors")
        self.assertEqual(self.nodes[14]["widgets_values"][-1], "all")

        lora_values = {
            node["title"]: node["widgets_values"]
            for node in self.workflow["nodes"]
            if node.get("type") == "FreeFuseLoRALoader"
        }
        self.assertEqual(
            lora_values["LoRA 1: Archer Queen"][:2],
            ["archer_queen_krea2.safetensors", "archer-queen"],
        )
        self.assertEqual(
            lora_values["LoRA 2: Satoru Gojo"][:2],
            ["satoru_gojo_krea2.safetensors", "satoru-gojo"],
        )

    def test_phase2_defaults_match_phase1_seed_and_steps(self):
        phase1_values = self.nodes[12]["widgets_values"]
        phase2_values = self.nodes[16]["widgets_values"]

        self.assertEqual(phase1_values[0], 8)
        self.assertEqual(phase1_values[1], 12)
        self.assertEqual(phase2_values[0], phase1_values[0])
        self.assertEqual(phase2_values[2], phase1_values[1])


if __name__ == "__main__":
    unittest.main()
