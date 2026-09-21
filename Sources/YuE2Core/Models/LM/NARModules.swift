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
    /// When the checkpoint's `pe` buffer was loaded (the common, macOS path), these are gathered
    /// from it verbatim — never recomputed to replace loaded values (pitfall #3: a mismatch there
    /// would be silent and wrong). When it was omitted (`computePE`, T-6.2's "-all" iPhone pack),
    /// the identical values are computed on demand for just these positions instead.
    public func latentPositions(count: Int) -> MLXArray {
        let maxIndex = Int32(config.maxLatentFrames - 1)
        let indices = MLX.minimum(MLXArray((0..<count).map { Int32($0) }), MLXArray(maxIndex))
        if let pe = latentPosEmbed.pe {
            return take(pe, indices, axis: 0)
        }
        return Self.computeLatentPositions(
            indices: indices, hiddenSize: config.hiddenSize, dtype: vae2llm.weight.dtype)
    }

    /// The checkpoint's own `AudioPositionEmbedding` formula (`modeling_yue2.py`), computed
    /// directly instead of gathered from a `[maxFrames, hiddenSize]` buffer:
    /// `pe[pos, 2i] = sin(pos·div_i)`, `pe[pos, 2i+1] = cos(pos·div_i)`,
    /// `div_i = exp(-ln(10000)·2i / hiddenSize)` — fp32 throughout, cast to `dtype` at the end
    /// (matching the reference, which builds its buffer in fp32 once at init).
    static func computeLatentPositions(indices: MLXArray, hiddenSize: Int, dtype: DType) -> MLXArray {
        let half = hiddenSize / 2
        let divTerm = MLX.exp(
            MLXArray(Array(stride(from: 0, to: hiddenSize, by: 2)).map { Float($0) })
                * (-Foundation.log(Float(10_000)) / Float(hiddenSize))
        )
        let angles = indices.asType(.float32).expandedDimensions(axis: -1) * divTerm.reshaped([1, half])
        let sinPart = MLX.sin(angles)
        let cosPart = MLX.cos(angles)
        // Interleave [sin0, cos0, sin1, cos1, ...] to match pe[:, 0::2]=sin, pe[:, 1::2]=cos.
        let interleaved = MLX.stacked([sinPart, cosPart], axis: -1).reshaped([indices.dim(0), hiddenSize])
        return interleaved.asType(dtype)
    }
}
