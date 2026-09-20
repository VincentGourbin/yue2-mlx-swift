"""protocol.json + tokenizer_corpus.json (plan/08-tests.md): exact tiktoken ids for the
real checkpoint-native prompt construction, generated with the upstream YuE2TextTokenizer/
token_prefixes/negative_prefix. Uses the real qwen.tiktoken (151643 ordinary tokens) —
$YUE2_MODELS_DIR/YuE2-3B/qwen.tiktoken if already downloaded (T-1.4), else fetched once
via hf_hub_download into the normal huggingface_hub cache (never committed)."""
import json
import os
from pathlib import Path

from yue2.protocol import SongRequest, negative_prefix, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer

_REFERENCE_EXAMPLES = Path(__file__).resolve().parents[2] / "reference" / "yue" / "examples"


def _qwen_tiktoken_path():
    models_dir = os.environ.get("YUE2_MODELS_DIR")
    if models_dir:
        candidate = Path(models_dir) / "YuE2-3B" / "qwen.tiktoken"
        if candidate.is_file():
            return candidate
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("m-a-p/YuE2-3B", "qwen.tiktoken"))


def _tonight_awake():
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download("m-a-p/YuE2-3B", "examples/tonight-awake.json")).read_text())


def build_protocol(tokenizer):
    song = json.loads((_REFERENCE_EXAMPLES / "song.json").read_text())
    tonight = _tonight_awake()
    abc_text = (_REFERENCE_EXAMPLES / "score.abc").read_text()
    entries = []
    for source_name, source in (("song", song), ("tonight-awake", tonight)):
        for cot in ("off", "melody", "full"):
            for abc_name, abc in (("none", None), ("score.abc", abc_text)):
                if abc is not None and cot == "off":
                    continue  # protocol.py forbids external ABC with cot=off.
                request = SongRequest(style=source["style"], lyrics=source["lyrics"], cot=cot, abc=abc)
                abc_ids = tokenizer.encode(abc) if abc is not None else None
                entry = {
                    "source": source_name, "cot": cot, "abc": abc_name,
                    "text": request.text(),
                    "prefix": token_prefixes(request, tokenizer, abc_ids),
                }
                # A negative (CFG) prefix needs the exact positive-branch ABC ids; those
                # only exist here for cot=off (no ABC at all) or when ABC was supplied.
                if cot == "off" or abc is not None:
                    entry["negative"] = negative_prefix(request, tokenizer, abc_ids)
                else:
                    entry["negative"] = None
                entries.append(entry)
    return entries


def build_tokenizer_corpus(tokenizer):
    song = json.loads((_REFERENCE_EXAMPLES / "song.json").read_text())
    tonight = _tonight_awake()
    texts = [
        SongRequest(style=song["style"], lyrics=song["lyrics"], cot="full").text(),
        (_REFERENCE_EXAMPLES / "score.abc").read_text(),
        (_REFERENCE_EXAMPLES / "score-jazz.abc").read_text(),
        (_REFERENCE_EXAMPLES / "melody.abc").read_text(),
        tonight["lyrics"],
        "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
        "I'LL  <|im_start|> <abc> x:1\nK:C\n",
        "café — naïve 🎵\n\n\n  deux  espaces",
    ]
    return [{"text": text, "ids": tokenizer.encode(text)} for text in texts]


def save(protocol_path="parity/protocol.json", corpus_path="parity/tokenizer_corpus.json"):
    tokenizer = YuE2TextTokenizer(_qwen_tiktoken_path())
    Path(protocol_path).write_text(json.dumps(build_protocol(tokenizer), ensure_ascii=False, indent=1) + "\n")
    Path(corpus_path).write_text(json.dumps(build_tokenizer_corpus(tokenizer), ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    save()
