#!/usr/bin/env python3
"""Static integrity checks for serialized ComfyUI workflows."""

import json
from collections import Counter
from pathlib import Path
import unittest


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflows"


def _normalise_type(value):
    return tuple(value) if isinstance(value, list) else value


class WorkflowIntegrityTests(unittest.TestCase):
    def test_ui_workflow_links_are_internally_consistent(self):
        workflow_paths = sorted(WORKFLOW_DIR.glob("*.json"))
        self.assertTrue(workflow_paths, "No workflow JSON files found")

        checked_ui_workflows = 0
        for workflow_path in workflow_paths:
            with self.subTest(workflow=workflow_path.name):
                with workflow_path.open("r", encoding="utf-8") as handle:
                    workflow = json.load(handle)

                # API-format prompts use a node-id mapping and do not contain
                # LiteGraph's top-level link metadata.
                if not isinstance(workflow.get("nodes"), list):
                    continue

                checked_ui_workflows += 1
                nodes = {node["id"]: node for node in workflow["nodes"]}
                link_records = workflow.get("links", [])
                link_ids = [record[0] for record in link_records]
                links = {record[0]: record for record in link_records}

                duplicate_ids = [
                    link_id
                    for link_id, count in Counter(link_ids).items()
                    if count > 1
                ]
                self.assertEqual(duplicate_ids, [], "Duplicate top-level link IDs")

                if link_ids:
                    self.assertGreaterEqual(
                        workflow["last_link_id"],
                        max(link_ids),
                        "last_link_id is behind an existing link ID",
                    )
                if nodes:
                    self.assertGreaterEqual(
                        workflow["last_node_id"],
                        max(nodes),
                        "last_node_id is behind an existing node ID",
                    )

                for node_id, node in nodes.items():
                    for output_slot, output in enumerate(node.get("outputs") or []):
                        for link_id in output.get("links") or []:
                            self.assertIn(
                                link_id,
                                links,
                                f"Node {node_id} output {output_slot} has a dangling link",
                            )
                            record = links[link_id]
                            self.assertEqual(
                                (record[1], record[2]),
                                (node_id, output_slot),
                                f"Link {link_id} has the wrong source endpoint",
                            )

                    for input_slot, input_data in enumerate(node.get("inputs") or []):
                        link_id = input_data.get("link")
                        if link_id is None:
                            continue
                        self.assertIn(
                            link_id,
                            links,
                            f"Node {node_id} input {input_slot} has a dangling link",
                        )
                        record = links[link_id]
                        self.assertEqual(
                            (record[3], record[4]),
                            (node_id, input_slot),
                            f"Link {link_id} has the wrong target endpoint",
                        )

                for link_id, record in links.items():
                    _, source_id, source_slot, target_id, target_slot, link_type = record[:6]
                    self.assertIn(source_id, nodes, f"Link {link_id} source node is missing")
                    self.assertIn(target_id, nodes, f"Link {link_id} target node is missing")

                    source_outputs = nodes[source_id].get("outputs") or []
                    target_inputs = nodes[target_id].get("inputs") or []
                    self.assertLess(
                        source_slot,
                        len(source_outputs),
                        f"Link {link_id} source slot is missing",
                    )
                    self.assertLess(
                        target_slot,
                        len(target_inputs),
                        f"Link {link_id} target slot is missing",
                    )

                    source = source_outputs[source_slot]
                    target = target_inputs[target_slot]
                    self.assertIn(
                        link_id,
                        source.get("links") or [],
                        f"Link {link_id} is absent from its source output",
                    )
                    self.assertEqual(
                        target.get("link"),
                        link_id,
                        f"Link {link_id} is absent from its target input",
                    )
                    self.assertEqual(
                        _normalise_type(source.get("type")),
                        _normalise_type(link_type),
                        f"Link {link_id} does not match its source slot type",
                    )
                    self.assertEqual(
                        _normalise_type(target.get("type")),
                        _normalise_type(link_type),
                        f"Link {link_id} does not match its target slot type",
                    )

        self.assertGreater(checked_ui_workflows, 0, "No UI-format workflows checked")


if __name__ == "__main__":
    unittest.main()
