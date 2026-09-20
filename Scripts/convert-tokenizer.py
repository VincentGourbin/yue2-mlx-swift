"""Converts Qwen2.5-0.5B's `tokenizer.json` into YuE2-3B's (fiche T-1.5).

Per plan/02.7-tokenizer.md, Qwen/Qwen2.5-0.5B's tokenizer.json has the exact same
vocabulary, merges, pre-tokenizer regex and NFC normalizer as the checkpoint-native
`qwen.tiktoken` (verified 2026-09-16, 0 gap over 151643 ranks). The only difference is
that it recognizes strings like "<|im_start|>" as special tokens; upstream's
`encode_ordinary` never does. This script empties `added_tokens` and disables
`post_processor` so the converted file matches upstream's behaviour exactly, then
verifies token-for-token equality against `parity/tokenizer_corpus.json` (ids already
computed via tiktoken in T-1.3).

It also copies Qwen2.5-0.5B's `tokenizer_config.json` alongside: swift-transformers'
`AutoTokenizer.from(modelFolder:)` needs a `tokenizer_class` to pick a tokenizer
implementation, and YuE2-3B's own config.json has a custom `model_type` ("yue2") with no
built-in fallback — this is plumbing, not a tokenizer behaviour change (swift-transformers
only reads `added_tokens` from tokenizer.json for its special-token set, never
`tokenizer_config.json`'s `added_tokens_decoder`).

Usage: `.venv-ref/bin/python Scripts/convert-tokenizer.py --models-dir "$YUE2_MODELS_DIR"`
"""
import argparse
import json
import sys
from pathlib import Path

QWEN_REPO = "Qwen/Qwen2.5-0.5B"


def fetch_qwen_files() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download
    tokenizer_json = Path(hf_hub_download(QWEN_REPO, "tokenizer.json"))
    tokenizer_config = Path(hf_hub_download(QWEN_REPO, "tokenizer_config.json"))
    return tokenizer_json, tokenizer_config


def convert(models_dir: Path) -> Path:
    tokenizer_json, tokenizer_config = fetch_qwen_files()
    data = json.loads(tokenizer_json.read_text())
    data["added_tokens"] = []
    data["post_processor"] = None
    dest_dir = models_dir / "YuE2-3B"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "tokenizer.json"
    dest.write_text(json.dumps(data, ensure_ascii=False))
    (dest_dir / "tokenizer_config.json").write_text(tokenizer_config.read_text())
    return dest


def verify(models_dir: Path) -> int:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(models_dir / "YuE2-3B" / "tokenizer.json"))
    corpus = json.loads(Path("parity/tokenizer_corpus.json").read_text())
    for i, item in enumerate(corpus):
        ids = tokenizer.encode(item["text"], add_special_tokens=False).ids
        if ids != item["ids"]:
            print(f"TOKENIZER MISMATCH at corpus[{i}]: {item['text'][:80]!r}")
            print(f"  expected ({len(item['ids'])}): {item['ids'][:20]}...")
            print(f"  got      ({len(ids)}): {ids[:20]}...")
            return 1
    print(f"TOKENIZER OK {len(corpus)} textes")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    args = parser.parse_args()
    models_dir = Path(args.models_dir)
    convert(models_dir)
    sys.exit(verify(models_dir))
