// AudioImporter.swift - read an arbitrary audio file into [1,S,2] fp32 (T-5.2)
// Copyright 2026 Vincent Gourbin
//
// Mirrors AudioExporter.swift's role in reverse. Uses AVFoundation (not a hand-rolled decoder)
// so any format the system can decode (MP3, WAV, M4A, ...) works; AVAudioConverter handles both
// resampling and channel remixing (mono/multichannel -> stereo) without any manual mixing code.

import AVFoundation
import Foundation
@preconcurrency import MLX

public enum AudioImporter {
    /// Reads `url`, converts to fp32 stereo at `sampleRate`, and returns `[1, S, 2]` (NLC --
    /// `YuE2VAEEncoder.encode`'s expected input layout).
    public static func loadAudio(url: URL, sampleRate: Double = 48_000) throws -> MLXArray {
        guard let file = try? AVAudioFile(forReading: url) else {
            throw YuE2Error.missingFile(url.path)
        }
        guard file.length > 0 else {
            throw YuE2Error.invalidRequest("\(url.lastPathComponent): empty audio file")
        }
        let sourceFormat = file.processingFormat
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 2, interleaved: false
        ) else {
            throw YuE2Error.invalidRequest("could not build a stereo float32 @ \(sampleRate) Hz target format")
        }
        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw YuE2Error.invalidRequest("\(url.lastPathComponent): cannot convert \(sourceFormat) to \(targetFormat)")
        }
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw YuE2Error.invalidRequest("could not allocate a \(file.length)-frame input buffer")
        }
        try file.read(into: inputBuffer)

        // Read-the-whole-file-once conversion: the input block hands over the single source
        // buffer, then reports end-of-stream. Output capacity is sized generously (target rate
        // can only ever produce ceil(inputFrames * ratio) frames, plus slack for converter
        // priming) and trimmed to `outputBuffer.frameLength` afterward.
        let ratio = sampleRate / sourceFormat.sampleRate
        let estimatedFrames = AVAudioFrameCount((Double(inputBuffer.frameLength) * ratio).rounded(.up)) + 4096
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: estimatedFrames) else {
            throw YuE2Error.invalidRequest("could not allocate a \(estimatedFrames)-frame output buffer")
        }

        var inputConsumed = false
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if inputConsumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            inputConsumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }
        if status == .error || conversionError != nil {
            throw YuE2Error.invalidRequest("\(url.lastPathComponent): conversion failed: \(conversionError?.localizedDescription ?? "unknown")")
        }
        let frameCount = Int(outputBuffer.frameLength)
        guard frameCount > 0, let channelData = outputBuffer.floatChannelData else {
            throw YuE2Error.invalidRequest("\(url.lastPathComponent): conversion produced no audio")
        }

        let left = channelData[0]
        let right = channelData[1]
        var interleaved = [Float](repeating: 0, count: frameCount * 2)
        for i in 0..<frameCount {
            interleaved[i * 2] = left[i]
            interleaved[i * 2 + 1] = right[i]
        }
        return MLXArray(interleaved, [1, frameCount, 2])
    }
}
