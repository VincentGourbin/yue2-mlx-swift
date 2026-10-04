// SheetSage2Transcriber.swift - whole-song transcription with overlapping windows (pipeline_sheetsage2.py port)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// One window of the sliding plan (`sliding_window_plan`).
struct TranscriptionWindow {
    var start: Double
    var end: Double
    var acceptStart: Double
    var acceptEnd: Double
    var prefixEnd: Double
    var generationStop: Double?
}

/// Where a transcription is, passed to `SheetSage2Transcriber.checkpoint` before every unit of GPU
/// work.
public struct SheetSage2Progress: Sendable, Equatable {
    public enum Stage: String, Sendable { case encoding, decoding }
    public var stage: Stage
    /// 0-based window index and window count (one window per 300 s, overlapping).
    public var window: Int
    public var windows: Int
    /// Encoding: fraction of this window's encoder already evaluated, in [0, 1).
    public var encoderFraction: Double
    /// Decoding: tokens of this window so far (prefix included).
    public var tokens: Int
    /// Rough whole-run fraction for a progress bar: encoding counts for a quarter, decoding for the
    /// rest (its length is not known in advance, so a window's decoding is credited once done).
    public var overallFraction: Double {
        let w = Double(max(windows, 1))
        switch stage {
        case .encoding: return 0.25 * (Double(window) + encoderFraction) / w
        case .decoding: return 0.25 + 0.75 * Double(window) / w
        }
    }
}

/// Transcription result: ABC (or why it could not be built), timed events, raw tokens per window.
public struct SheetSage2Transcription: Sendable {
    public var abc: String?
    public var abcError: String?
    public var events: [SheetSage2Event]
    public var tokens: [[Int]]
    public var durationSeconds: Double
    public var warnings: [String]
}

public struct SheetSage2Transcriber {
    public let model: SheetSage2Model
    public var overlapSeconds = 200.0
    public var lookaheadSeconds = 100.0
    /// MLX buffer-cache limit while transcribing (restored afterwards). Without it the cache keeps
    /// every freed encoder transient: +3 GB of footprint on a 300 s window. `nil` leaves it as is.
    public var cacheLimitBytes: Int? = 512 * 1024 * 1024
    /// Release the encoder's weights once every window is encoded (lean profiles). The model
    /// must then be loaded again before another transcription.
    public var releaseEncoderAfterEncoding = false
    /// Called before every unit of GPU work (an encoder stage or chunk, a decoding step): nothing is
    /// submitted to the GPU between two calls. Block in it to pause — an iOS app waits on its GPU
    /// gate here while it is not in the foreground — and throw from it to cancel; the error is
    /// rethrown by `transcribe`. A cancelled `Task` is also checked at the same points
    /// (`CancellationError`).
    public var checkpoint: ((SheetSage2Progress) throws -> Void)?

    public init(model: SheetSage2Model) {
        self.model = model
    }

    static func windowPlan(duration: Double, window: Double, overlap: Double, lookahead: Double) -> [TranscriptionWindow] {
        let hop = window - overlap
        var (start, accepted) = (0.0, 0.0)
        var result = [TranscriptionWindow]()
        while true {
            let last = start + window >= duration - 1e-6
            let acceptEnd = last ? duration : start + window - lookahead
            result.append(TranscriptionWindow(
                start: start, end: min(duration, start + window), acceptStart: accepted, acceptEnd: acceptEnd,
                prefixEnd: accepted, generationStop: last ? nil : window - lookahead))
            if last { return result }
            accepted = acceptEnd
            start = min(start + hop, duration - window)
        }
    }

    /// Transcribes a 24 kHz mono waveform (`[samples]`, not normalized).
    public func transcribe(
        _ waveform: MLXArray, melodyOnly: Bool = true, onToken: ((Int, Int) -> Void)? = nil
    ) throws -> SheetSage2Transcription {
        let previousCacheLimit = Memory.cacheLimit
        if let cacheLimitBytes { Memory.cacheLimit = cacheLimitBytes }
        defer { if cacheLimitBytes != nil { Memory.cacheLimit = previousCacheLimit } }
        let tokenizer = SheetSage2Tokenizer(audioLengthSeconds: model.config.inputAudioLength, timeHz: model.config.timeHz)
        let prompts = SheetSage2Tokenizer.fullTaskPrompts
        let rate = Double(model.config.samplingRate)
        let samples = waveform.dim(0)
        let duration = Double(samples) / rate
        let windowLength = model.config.inputAudioLength
        let plan = Self.windowPlan(duration: duration, window: windowLength, overlap: overlapSeconds, lookahead: lookaheadSeconds)
        // Every window is encoded before any decoding (windows only depend on the audio), so the
        // encoder can be released before the decoding loop.
        let hook = checkpoint
        func check(_ progress: SheetSage2Progress) throws {
            if Task.isCancelled { throw CancellationError() }
            try hook?(progress)
        }
        var memories = [MLXArray]()
        for (index, window) in plan.enumerated() {
            let offset = Int((window.start * rate).rounded())
            let count = Int((windowLength * rate).rounded())
            memories.append(try model.encode(waveform[offset ..< min(samples, offset + count)]) { fraction in
                try check(SheetSage2Progress(stage: .encoding, window: index, windows: plan.count, encoderFraction: fraction, tokens: 0))
            })
            Memory.clearCache()
        }
        if releaseEncoderAfterEncoding { model.releaseEncoder() }
        var generator = SheetSage2Generator(model: model)
        var stitched = [SheetSage2Event]()
        var allTokens = [[Int]]()
        var warnings = [String]()
        for (index, window) in plan.enumerated() {
            var prefix: [Int]?
            var base = 0
            if index > 0, let built = OverlapPrefix.build(
                stitched: stitched, tokenizer: tokenizer, prompts: prompts, windowStart: window.start, prefixEnd: window.prefixEnd)
            {
                guard built.tokens.count < model.config.maxOutputSeqLen - 128 else {
                    throw SheetSage2Error.invalidSequence("Overlap prefix fills the context; reduce overlap_seconds")
                }
                (prefix, base) = (built.tokens, built.baseSubbeat)
            }
            let memory = memories[index]
            let stop = window.generationStop ?? min(duration - window.start, windowLength)
            let tokens = try generator.generate(
                memory: memory, prompts: prompts, prefixTokens: prefix, stopTimeSeconds: stop,
                onToken: { onToken?(index, $0) },
                checkpoint: { tokens in
                    try check(SheetSage2Progress(stage: .decoding, window: index, windows: plan.count, encoderFraction: 1, tokens: tokens))
                })
            if tokens.count > model.config.maxOutputSeqLen {
                warnings.append("Window \(index + 1) reached the token limit; inspect its token coverage")
            }
            let decoded = try tokenizer.decodeGenerated(tokens)
            if let warning = decoded.warning { warnings.append(warning) }
            let timeMap = EventTimeMap(events: decoded.events, targetSeconds: windowLength)
            stitched += stitchedWindowEvents(
                decoded.events, timeMap: timeMap, windowStart: window.start, acceptStart: window.acceptStart,
                acceptEnd: window.acceptEnd, songDuration: duration, windowIndex: index, globalSubbeatBase: base)
            allTokens.append(tokens)
            Memory.clearCache()
        }
        stitched.sort { ($0.time!, $0.globalSubbeat) < ($1.time!, $1.globalSubbeat) }
        var abc: String?
        var abcError: String?
        do {
            abc = try AbcNotation.abc(events: stitched, duration: duration, melodyOnly: melodyOnly)
        } catch {
            abcError = String(describing: error)
        }
        return SheetSage2Transcription(
            abc: abc, abcError: abcError, events: stitched, tokens: allTokens, durationSeconds: duration, warnings: warnings)
    }
}

/// `build_overlap_prefix_tokens`: the previous windows' events inside this window's overlap,
/// re-encoded as the decoder's prefix so the new window continues them.
enum OverlapPrefix {
    static func build(
        stitched: [SheetSage2Event], tokenizer: SheetSage2Tokenizer, prompts: [String], windowStart: Double, prefixEnd: Double
    ) -> (tokens: [Int], baseSubbeat: Int)? {
        let eps = 1e-4
        var source = stitched.filter { windowStart - eps <= ($0.time ?? -1) && ($0.time ?? -1) < prefixEnd - eps }
        source.sort { ($0.globalSubbeat, $0.time ?? 0) < ($1.globalSubbeat, $1.time ?? 0) }
        guard let first = source.firstIndex(where: { $0.values.timestamp != nil || $0.values.hasRhythm }) else { return nil }
        source = Array(source[first...])
        let base = source[0].globalSubbeat
        let context = activeContext(before: source[0].time!, in: stitched, tokenizer: tokenizer)
        var events = source.map { event -> SheetSage2Event in
            var copy = event
            copy.subbeat = max(0, event.globalSubbeat - base)
            if copy.tokensByField["timestamp"] != nil {
                let id = max(0, min(Int(((event.time! - windowStart) * Double(tokenizer.timeHz)).rounded(.toNearestOrEven)), tokenizer.nTimeTokens - 1))
                copy.tokensByField["timestamp"] = [tokenizer.timeTokenStart + id]
            }
            return copy
        }
        for field in ["structure", "key", "chord"] where events[0].tokensByField[field] == nil {
            if let tokens = context[field] { events[0].tokensByField[field] = tokens }
        }
        let rhythm = events[0].tokensByField["rhythm"] ?? []
        if rhythm.contains(where: { tokenizer.tokenType($0) == .eighthPosition }),
            !rhythm.contains(where: { tokenizer.tokenType($0) == .meter }), let meter = context["meter"]
        {
            events[0].tokensByField["rhythm"] = meter + rhythm
        }
        // encode_decoded_sequence, has_eos=False.
        var tokens = tokenizer.promptPrefix(prompts)
        var previous = 0
        for event in events {
            var shift = event.subbeat - previous
            while shift > 256 {
                tokens.append(tokenizer.subbeatShiftTokenStart + 256)
                shift -= 256
            }
            tokens.append(tokenizer.subbeatShiftTokenStart + shift)
            previous = event.subbeat
            for field in SheetSage2Tokenizer.eventFieldOrder { tokens += event.tokensByField[field] ?? [] }
        }
        return (tokens, base)
    }

    static func activeContext(before time: Double, in events: [SheetSage2Event], tokenizer: SheetSage2Tokenizer) -> [String: [Int]] {
        var state = [String: [Int]]()
        for event in events {
            guard let eventTime = event.time, eventTime <= time + 1e-6 else { continue }
            for field in ["structure", "key", "chord"] {
                if let tokens = event.tokensByField[field], !tokens.isEmpty { state[field] = tokens }
            }
            let meters = (event.tokensByField["rhythm"] ?? []).filter { tokenizer.tokenType($0) == .meter }
            if let first = meters.first { state["meter"] = [first] }
        }
        return state
    }
}
