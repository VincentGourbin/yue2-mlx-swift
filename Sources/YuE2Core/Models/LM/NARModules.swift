// NARModules.swift - NAR-side scalar/embedding math: timestep shift, sinusoidal embedding,
// latent position lookup (plan §2.4, pitfalls #3 and #4)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

extension YuE2ForCausalLM {
    /// `shift · sigmoid(t) / (1 + (shift-1) · sigmoid(t))`. The sigmoid itself is computed in the
    /// model's compute dtype (bf16 in real weights) — pitfall #4: doing it in fp32 and casting
    /// down afterwards is a different (wrong) number. With the checkpoint's `timestep_shift == 1`
    /// this collapses to `sigmoid(t)`, but the general formula is kept to match the reference.
    public func shiftT(raw: Double) -> MLXArray {
        let dtype = vae2llm.weight.dtype
        let tSig = MLX.sigmoid(MLXArray(Float(raw)).asType(dtype))
        let shift = config.timestepShift
        if shift == 1 {
            return tSig
        }
        let shiftArray = MLXArray(shift).asType(dtype)
        let one = MLXArray(Float(1)).asType(dtype)
        return shiftArray * tSig / (one + (shiftArray - one) * tSig)
    }

    /// Sinusoidal timestep embedding (`frequency_embedding_size = 256`, computed in fp32) through
    /// `time_embedder`'s MLP. `tShifted` may carry one value (bisection fixtures) or one value per
    /// NAR position (`velocity`, where the same scalar is broadcast across the chunk) — the
    /// sinusoid and MLP are applied per-element either way, returning `[tShifted.count, hiddenSize]`.
    public func timeEmbedding(tShifted: MLXArray) -> MLXArray {
        let half = 128
        let freqs = MLX.exp(
            MLXArray(Array(0..<half)).asType(.float32) * (-Foundation.log(Float(10_000)) / Float(half))
        )
        let args = tShifted.asType(.float32).expandedDimensions(axis: -1) * freqs.reshaped([1, half])
        let embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
            .asType(vae2llm.weight.dtype)
        return timeEmbedder.fc2(silu(timeEmbedder.fc1(embedding)))
    }

    /// `latent_pos_embed.pe[clamp(arange(count), max: maxLatentFrames-1)]` — each index is
    /// clamped individually (not the output truncated), matching the reference's
    /// `torch.arange(nar_length).clamp(max=max_latent_frames-1)` gather. In practice a NAR
    /// chunk's length never approaches `maxLatentFrames`, so this only matters defensively.
    /// The table itself is loaded verbatim from the checkpoint, never recomputed (pitfall #3).
    public func latentPositions(count: Int) -> MLXArray {
        let maxIndex = Int32(config.maxLatentFrames - 1)
        let indices = MLX.minimum(MLXArray((0..<count).map { Int32($0) }), MLXArray(maxIndex))
        return take(latentPosEmbed.pe, indices, axis: 0)
    }
}
