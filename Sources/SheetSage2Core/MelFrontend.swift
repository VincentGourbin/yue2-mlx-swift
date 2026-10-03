// MelFrontend.swift - MERT2 power log-mel features (plan/14-sheetsage2.md, fact 1)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// torchaudio `Spectrogram(power=2, center=True, pad_mode="reflect")` → `MelScale` →
/// `AmplitudeToDB(top_db=None)`, last frame dropped, then the checkpoint's fixed normalization.
/// The Hann window and the mel filter bank are the checkpoint's own buffers, so they are exact by
/// construction. Always float32, whatever the model's compute dtype.
final class MelFrontend: Module {
    let nFFT: Int
    let hopLength: Int

    @ParameterInfo(key: "mel_mean") var melMean: MLXArray
    @ParameterInfo(key: "mel_std") var melStd: MLXArray
    @ModuleInfo(key: "spectrogram") var spectrogram: SpectrogramBuffers
    @ModuleInfo(key: "mel_scale") var melScale: MelScaleBuffers

    init(config: MERT2Config) {
        nFFT = config.nFFT
        hopLength = config.hopLength
        _melMean.wrappedValue = MLXArray.zeros([config.numMelBins])
        _melStd.wrappedValue = MLXArray.ones([config.numMelBins])
        _spectrogram.wrappedValue = SpectrogramBuffers(winLength: config.winLength)
        _melScale.wrappedValue = MelScaleBuffers(nSTFT: config.nFFT / 2 + 1, nMels: config.numMelBins)
        super.init()
    }

    /// `waveform` `[samples]` float32 → `[frames − 1, nMels]` normalized log-mel.
    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        let x = waveform.asType(.float32)
        let half = nFFT / 2
        // Reflect padding (torch.stft center=True): x[half], …, x[1] | x | x[n−2], …, x[n−1−half].
        let n = x.dim(0)
        let left = x[MLXArray(Array(stride(from: half, to: 0, by: -1)).map(Int32.init))]
        let right = x[MLXArray(Array(stride(from: n - 2, to: n - 2 - half, by: -1)).map(Int32.init))]
        let padded = concatenated([left, x, right], axis: 0)
        let frames = 1 + (padded.dim(0) - nFFT) / hopLength
        let windowed = asStrided(padded, [frames, nFFT], strides: [hopLength, 1]) * spectrogram.window
        let spectrum = MLX.abs(MLXFFT.rfft(windowed, axis: -1))
        let power = spectrum * spectrum
        let mel = matmul(power, melScale.fb)
        let db = 10 * MLX.log10(MLX.maximum(mel, MLXArray(Float(1e-10))))
        let trimmed = db[0..<(frames - 1)]
        return (trimmed - melMean) / MLX.maximum(melStd, MLXArray(Float(1e-5)))
    }
}

/// Holder for torchaudio `Spectrogram`'s `window` buffer (key `spectrogram.window`).
final class SpectrogramBuffers: Module {
    @ParameterInfo(key: "window") var window: MLXArray
    init(winLength: Int) {
        _window.wrappedValue = MLXArray.zeros([winLength])
        super.init()
    }
}

/// Holder for torchaudio `MelScale`'s `fb` buffer `[nSTFT, nMels]` (key `mel_scale.fb`).
final class MelScaleBuffers: Module {
    @ParameterInfo(key: "fb") var fb: MLXArray
    init(nSTFT: Int, nMels: Int) {
        _fb.wrappedValue = MLXArray.zeros([nSTFT, nMels])
        super.init()
    }
}
