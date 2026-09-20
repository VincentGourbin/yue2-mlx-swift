// AudioExporter.swift - stereo WAV export (plan §1.6; adapted from ltx-video-swift-mlx)
// Copyright 2026 Vincent Gourbin

import Foundation
@preconcurrency import MLX

/// Exports a stereo waveform to a WAV file.
public enum AudioExporter {
    public enum SampleFormat {
        case int16
        case float32
    }

    /// - Parameters:
    ///   - audio: stereo waveform, `[2, T]` or `[B, 2, T]`, values expected in `[-1, 1]`.
    ///   - sampleRate: output sample rate in Hz.
    ///   - gain: linear gain applied before export (1.0 = no change).
    ///   - format: `.int16` (16-bit PCM) or `.float32` (32-bit IEEE float).
    public static func exportToWAV(
        audio: MLXArray,
        url: URL,
        sampleRate: Int = 48_000,
        gain: Float = 1.0,
        format: SampleFormat = .int16
    ) throws {
        var samples = gain != 1.0 ? audio * gain : audio
        if samples.ndim == 3 {
            samples = samples.squeezed(axis: 0) // [2, T]
        }
        let channels = samples.dim(0)
        let numFrames = samples.dim(1)

        // Clamp only at export time (plan §9 piège n°11): interior math stays unclamped.
        // Single eval materializes the full lazy graph before we walk it on the CPU.
        let clamped = MLX.clip(samples, min: -1.0, max: 1.0).asType(.float32)
        let interleaved = clamped.transposed(1, 0).reshaped([-1]) // [L0, R0, L1, R1, ...]
        MLX.eval(interleaved)
        let floatSamples = interleaved.asArray(Float.self)

        let bitsPerSample: UInt16
        let audioFormatCode: UInt16
        var payload: Data

        switch format {
        case .int16:
            bitsPerSample = 16
            audioFormatCode = 1 // WAVE_FORMAT_PCM
            payload = Data(capacity: floatSamples.count * 2)
            for sample in floatSamples {
                let scaled = Int32((sample * 32767.0).rounded())
                var value = Int16(max(-32768, min(32767, scaled))).littleEndian
                payload.append(Data(bytes: &value, count: 2))
            }
        case .float32:
            bitsPerSample = 32
            audioFormatCode = 3 // WAVE_FORMAT_IEEE_FLOAT
            payload = Data(capacity: floatSamples.count * 4)
            for sample in floatSamples {
                var le = sample.bitPattern.littleEndian
                payload.append(Data(bytes: &le, count: 4))
            }
        }

        let dataSize = UInt32(payload.count)
        let fileSize = 36 + dataSize
        let blockAlign = UInt16(channels) * (bitsPerSample / 8)
        let byteRate = UInt32(sampleRate) * UInt32(blockAlign)

        var header = Data(capacity: 44)
        header.append(contentsOf: "RIFF".utf8)
        appendLE(&header, fileSize)
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        appendLE(&header, UInt32(16))
        appendLE(&header, audioFormatCode)
        appendLE(&header, UInt16(channels))
        appendLE(&header, UInt32(sampleRate))
        appendLE(&header, byteRate)
        appendLE(&header, blockAlign)
        appendLE(&header, bitsPerSample)
        header.append(contentsOf: "data".utf8)
        appendLE(&header, dataSize)

        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var file = header
        file.append(payload)
        try file.write(to: url)

        YuE2Debug.log("Exported WAV: \(channels)ch, \(sampleRate)Hz, \(numFrames) frames, \(file.count) bytes -> \(url.path)")
    }

    private static func appendLE(_ data: inout Data, _ value: UInt16) {
        var le = value.littleEndian
        data.append(Data(bytes: &le, count: 2))
    }

    private static func appendLE(_ data: inout Data, _ value: UInt32) {
        var le = value.littleEndian
        data.append(Data(bytes: &le, count: 4))
    }
}
