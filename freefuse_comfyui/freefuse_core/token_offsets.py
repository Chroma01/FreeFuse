"""Byte offsets for the exact native Qwen token stream, including slow tokenizers."""


def native_byte_offsets(tokenizer, ids, text):
    raw = text.encode("utf-8")
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) != list(ids):
            raise ValueError("Tokenizer offsets do not reproduce native conditioning tokens.")
        return [(len(text[:a].encode("utf-8")), len(text[:b].encode("utf-8")))
                for a, b in encoded["offset_mapping"]]
    if not hasattr(tokenizer, "byte_decoder"):
        raise ValueError("Native token offsets require a fast tokenizer or a byte-level Qwen tokenizer.")
    joined = bytearray()
    offsets = []
    special = set(tokenizer.all_special_ids)
    for token_id, piece in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
        value = (piece.encode("utf-8") if token_id in special else
                 bytes(tokenizer.byte_decoder[c] for c in piece))
        offsets.append((len(joined), len(joined) + len(value)))
        joined.extend(value)
    if not joined.startswith(raw):
        raise ValueError("Native byte tokens do not reproduce the encoded prompt.")
    return offsets
