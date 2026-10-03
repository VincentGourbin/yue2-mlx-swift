"""SheetSage2 parity fixtures and weight conversion (plan/14-sheetsage2.md, step S-1).

Runs the upstream classes from `reference/sheetsage2` unmodified (environment:
`Scripts/setup-sheetsage2-env.sh`), never a reimplementation.

    .venv-sheetsage2/bin/python Scripts/reference/sheetsage2_fixtures.py tiny
    .venv-sheetsage2/bin/python Scripts/reference/sheetsage2_fixtures.py convert --models-dir "$YUE2_MODELS_DIR"
    .venv-sheetsage2/bin/python Scripts/reference/sheetsage2_fixtures.py real --models-dir "$YUE2_MODELS_DIR" \
        --audio NAME=path.wav [--audio NAME2=path2.wav]

`tiny`    random-weight miniature (seed 1407) -> parity/tiny_sheetsage2.safetensors (committed).
`convert` merges the LoRA adapters into MERT-v2 (upstream `from_pretrained`) and writes a standalone
          fp32 snapshot to $YUE2_MODELS_DIR/SheetSage2-merged (upstream `save_pretrained`), the parity
          reference for the Swift-side merge of `yue2 download --model sheetsage2`.
`song`    whole-song CPU fp32 reference with sliding windows -> sheetsage2_song_NAME.*
`real`    CPU fp32 run of the real model on each audio file -> $YUE2_MODELS_DIR/parity/sheetsage2_NAME.*
          (gitignored, never committed: the audio may be copyrighted).
"""
import argparse
import datetime
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "reference" / "sheetsage2"
REVISION = "398b22834dac7dd05e09b9c4e40a39fc479ec502"
sys.path.insert(0, str(REFERENCE.parent))  # import the upstream files as the `sheetsage2` package

META = {"upstream": "m-a-p/SheetSage2", "upstream_revision": REVISION,
        "generated": datetime.date.today().isoformat()}


def save_deterministic(tensors, path, metadata):
    """`save_file` with a canonical header (same reason as `_common.save_file_deterministic`)."""
    tensors = {k: v.detach().contiguous().clone() for k, v in tensors.items()}
    save_file(tensors, path, metadata=metadata)
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(canonical)))
        f.write(canonical)
        f.write(data)


def encoder_taps(model, waveform, tap_layers):
    """Re-run `get_audio_features` step by step, returning every intermediate the Swift port bisects."""
    out = {}
    padded, _ = model._prepare_audio(waveform[None])
    out["padded_waveform_samples"] = torch.tensor([padded.shape[-1]], dtype=torch.int64)
    mel = model.encoder.feature_extractor(padded)
    out["mel"] = mel[0]
    hidden = mel
    for i, block in enumerate(model.encoder.subsampling_module):
        hidden = block(hidden)
        out[f"subsampling_{i}"] = hidden[0]
    weights = torch.softmax(model.layer_weight, dim=0)
    mixed = hidden * weights[0]
    positions = model.encoder.embed_positions(hidden)
    for i, (weight, layer) in enumerate(zip(weights[1:], model.encoder.layers)):
        hidden = layer(hidden, positions)
        mixed = mixed + hidden * weight
        if i in tap_layers:
            out[f"layer_{i}"] = hidden[0]
    out["mixed"] = mixed[0]
    out["memory"] = model.encoder_projection(mixed)[0]
    reference = model.encode(waveform[None])[0]
    assert torch.equal(reference, out["memory"]), "step-by-step encoder differs from model.encode"
    return out


def decoder_taps(model, memory, tokens, positions):
    """Teacher-forced logits at chosen positions, plus the first cached steps."""
    out = {}
    ids = torch.tensor([tokens], dtype=torch.long)
    logits, _ = model.decode(memory[None], ids, use_cache=False)
    out["teacher_positions"] = torch.tensor(positions, dtype=torch.int64)
    out["teacher_logits"] = logits[0, positions]
    return out


# ---------------------------------------------------------------------------- tiny

def tiny():
    from sheetsage2.configuration_sheetsage2 import SheetSage2Config
    from sheetsage2.modeling_sheetsage2 import SheetSage2Model
    from sheetsage2.tokenization_sheetsage2 import SheetSage2Tokenizer

    torch.manual_seed(1407)
    seconds = 0.48  # 11 520 samples: 48 mel frames, 12 encoder frames, 48 time tokens
    tiny_tokenizer = SheetSage2Tokenizer(seconds, 100, "v1")
    vocab = tiny_tokenizer.n_tokens
    backbone = dict(
        architectures=["MERT2Model"], context_seconds=seconds, conv_depthwise_kernel_size=31,
        frame_rate=25.0, hidden_size=32, hop_length=240, initializer_range=0.02,
        inputs_to_logits_ratio=960, intermediate_size=64, layer_norm_eps=1e-5,
        minimum_input_samples=1025, model_type="mert2", n_fft=256, num_attention_heads=4,
        num_hidden_layers=2, num_mel_bins=16, rotary_embedding_base=10000, sampling_rate=24000,
        subsampling_channels=[16, 32, 32], subsampling_depths=[1, 1, 1],
        subsampling_layer_norm_eps=1e-6, variant="fs", win_length=256,
    )
    config = SheetSage2Config(
        backbone_config=backbone, decoder_layers=2, hidden_size=16, intermediate_size=32,
        num_attention_heads=2, input_audio_length=seconds, max_output_seq_len=64,
        vocab_size=vocab, weights_format="merged", tokenizer_fingerprint=tiny_tokenizer.vocab_fingerprint,
        lora_rank=4, lora_alpha=8.0,
    )
    model = SheetSage2Model(config).eval()
    with torch.no_grad():
        # post_init leaves LayerNorm at (1, 0), GRN and layer_weight at 0: perturb everything so
        # every parameter is exercised by the parity test (a zero GRN would hide a wrong axis).
        for name, param in model.named_parameters():
            param.add_(torch.randn_like(param) * (0.1 if param.ndim > 1 else 0.3))
        fe = model.encoder.feature_extractor
        fe.mel_mean.copy_(torch.randn_like(fe.mel_mean) * 5 - 40)
        fe.mel_std.copy_(torch.rand_like(fe.mel_std) * 10 + 5)
    out = {f"weights.{k}": v for k, v in model.state_dict().items()}
    with torch.inference_mode():
        waveform = torch.randn(7_000, generator=torch.Generator().manual_seed(7)) * 0.3
        out["input_waveform"] = waveform
        out.update({f"encoder.{k}": v for k, v in encoder_taps(model, waveform, {0, 1}).items()})
        memory = out["encoder.memory"]
        tokenizer = model.tokenizer
        tokens = tokenizer.prompt_prefix(("timestamp", "melody_full"))
        tokens += [tokenizer.subbeat_shift_token_start, tokenizer.time_token_start + 3,
                   tokenizer.pitch_token_start + 60, tokenizer.duration_token_start + 2,
                   tokenizer.subbeat_shift_token_start + 4, tokenizer.pitch_token_start + 62]
        out["decoder.tokens"] = torch.tensor(tokens, dtype=torch.int64)
        out.update({f"decoder.{k}": v for k, v in decoder_taps(model, memory, tokens, list(range(len(tokens)))).items()})
    meta = {**META, "kind": "tiny_sheetsage2", "seed": "1407", "config": json.dumps(config.to_dict(), sort_keys=True)}
    path = ROOT / "parity" / "tiny_sheetsage2.safetensors"
    save_deterministic(out, path, meta)
    print(f"tiny fixture written to {path} ({path.stat().st_size / 1e6:.2f} MB)")


# ------------------------------------------------------------------------- convert

def load_real():
    from transformers import AutoModel
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return AutoModel.from_pretrained(str(REFERENCE), trust_remote_code=True, local_files_only=True).eval()


def convert(models_dir: Path):
    model = load_real()
    destination = models_dir / "SheetSage2-merged"
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer = model.tokenizer
    vocab = {name: getattr(tokenizer, name) for name in dir(tokenizer)
             if name.endswith(("_token_start", "_token_end")) or name.endswith("_token")
             and isinstance(getattr(tokenizer, name), int)}
    vocab.update(n_tokens=tokenizer.n_tokens, fingerprint=tokenizer.vocab_fingerprint,
                 prompt_to_id=tokenizer.prompt_to_id)
    (destination / "vocabulary.json").write_text(json.dumps(vocab, indent=1, sort_keys=True))
    print(f"merged snapshot written to {destination}")


# ---------------------------------------------------------------------------- real

def real(models_dir: Path, audios):
    from sheetsage2.audio_sheetsage2 import load_audio
    model = load_real()
    parity = models_dir / "parity"
    parity.mkdir(parents=True, exist_ok=True)
    for name, path in audios:
        with torch.inference_mode():
            waveform = load_audio(path)
            out = {"input_waveform": waveform}
            out.update({f"encoder.{k}": v for k, v in encoder_taps(model, waveform, {0, 11, 23}).items()})
            steps = {}
            def capture(position, ids, logits, masked):
                if len(steps) < 4:
                    steps[position] = (logits[0].clone(), masked[0].clone())
            from sheetsage2.generation_sheetsage2 import constrained_prompt_generate, FULL_TASK_PROMPTS
            duration = len(waveform) / 24_000
            tokens = constrained_prompt_generate(
                model, waveform[None], FULL_TASK_PROMPTS, model.max_output_seq_len,
                autocast_dtype=None, stop_time_seconds=min(duration, 300.0),
                memory=out["encoder.memory"][None], step_callback=capture,
            ).tolist()
            out["decoder.tokens"] = torch.tensor(tokens, dtype=torch.int64)
            first = sorted(steps)
            out["decoder.first_step_positions"] = torch.tensor(first, dtype=torch.int64)
            out["decoder.first_step_logits"] = torch.stack([steps[p][0] for p in first])
            positions = sorted({len(tokens) // 4, len(tokens) // 2, len(tokens) - 2})
            out.update({f"decoder.{k}": v for k, v in decoder_taps(model, out["encoder.memory"], tokens, positions).items()})
            try:
                full = model.transcribe(waveform, sampling_rate=24_000, dtype="fp32", melody_only=True)
                melody_abc, melody_error = full["abc"], None
            except RuntimeError as error:
                melody_abc, melody_error = None, str(error)
            complete = model.transcribe(waveform, sampling_rate=24_000, dtype="fp32", melody_only=False)
            assert complete["tokens"][0].tolist() == tokens, "transcribe() tokens differ from the fixture run"
        save_deterministic(out, parity / f"sheetsage2_{name}.safetensors", {**META, "kind": "real_sheetsage2", "audio": Path(path).name})
        sidecar = dict(audio=Path(path).name, duration_seconds=duration, tokens=len(tokens),
                       abc_melody_only=melody_abc, abc_melody_only_error=melody_error,
                       abc_full=complete["abc"], events=complete["events"])
        (parity / f"sheetsage2_{name}.json").write_text(json.dumps(sidecar, indent=1, default=str))
        print(f"{name}: {len(tokens)} tokens, fixture in {parity}")


def song(models_dir: Path, audios):
    """Whole-song reference (sliding windows with overlap prefixes): tokens per window and ABC."""
    from sheetsage2.audio_sheetsage2 import load_audio
    model = load_real()
    parity = models_dir / "parity"
    parity.mkdir(parents=True, exist_ok=True)
    for name, path in audios:
        with torch.inference_mode():
            waveform = load_audio(path)
            result = model.transcribe(waveform, sampling_rate=24_000, dtype="fp32", melody_only=False)
            try:
                melody = model.transcribe(waveform, sampling_rate=24_000, dtype="fp32", melody_only=True)["abc"]
            except RuntimeError as error:
                melody = None
        save_deterministic({"input_waveform": waveform}, parity / f"sheetsage2_song_{name}.safetensors",
                           {**META, "kind": "song_sheetsage2", "audio": Path(path).name})
        sidecar = dict(audio=Path(path).name, duration_seconds=len(waveform) / 24_000,
                       window_tokens=[t.tolist() for t in result["tokens"]],
                       abc_full=result["abc"], abc_melody_only=melody)
        (parity / f"sheetsage2_song_{name}.json").write_text(json.dumps(sidecar))
        print(f"{name}: {len(result['tokens'])} windows, {[len(t) for t in result['tokens']]} tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("tiny", "convert", "real", "song"))
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--audio", action="append", default=[], help="NAME=path")
    args = parser.parse_args()
    if args.command == "tiny":
        tiny()
    elif args.command == "convert":
        convert(args.models_dir)
    elif args.command == "real":
        real(args.models_dir, [tuple(item.split("=", 1)) for item in args.audio])
    else:
        song(args.models_dir, [tuple(item.split("=", 1)) for item in args.audio])


if __name__ == "__main__":
    main()
