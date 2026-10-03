// PromptGrammar.swift - grammar-constrained greedy decoding (plan/14-sheetsage2.md, S-4)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// Port of `PromptGrammarState`: which token blocks may follow, given the event being built.
struct PromptGrammarState: Hashable {
    enum Incomplete: Hashable { case rhythmAfterMeter, melodyAfterPitch }

    static let fieldIndex = ["timestamp": 0, "rhythm": 1, "structure": 2, "key": 3, "chord": 4, "melody": 5]

    var inShift = true
    var shiftRun = 0
    var payloadCount = 0
    var lastFieldIndex = -1
    var incomplete: Incomplete?

    /// The part of the state `allowed()` reads; masks are cached on it.
    struct MaskKey: Hashable {
        let canEnd: Bool
        let allowShift: Bool
        let incomplete: Incomplete?
        let lastFieldIndex: Int
    }

    var maskKey: MaskKey {
        MaskKey(
            canEnd: payloadCount > 0, allowShift: (payloadCount > 0 || inShift) && shiftRun < 4,
            incomplete: incomplete, lastFieldIndex: lastFieldIndex)
    }

    /// Allowed token ranges for `key` (`allowed()` upstream).
    static func allowedRanges(_ key: MaskKey, _ t: SheetSage2Tokenizer) -> [Range<Int>] {
        var ranges = [Range<Int>]()
        if key.canEnd { ranges.append(t.eosToken ..< t.eosToken + 1) }
        if key.allowShift { ranges.append(t.subbeatShiftTokenStart ..< t.subbeatShiftTokenEnd) }
        switch key.incomplete {
        case .rhythmAfterMeter:
            ranges.append(t.eighthPositionTokenStart ..< t.eighthPositionTokenEnd)
            return ranges
        case .melodyAfterPitch:
            ranges.append(t.durationTokenStart ..< t.durationTokenEnd)
            ranges.append(t.pitchTokenStart ..< t.pitchTokenEnd)
            return ranges
        case nil:
            break
        }
        let last = key.lastFieldIndex
        if last < 0 { ranges.append(t.timeTokenStart ..< t.timeTokenEnd) }
        if last < 1 {
            ranges.append(t.meterTokenStart ..< t.meterTokenEnd)
            ranges.append(t.eighthPositionTokenStart ..< t.eighthPositionTokenEnd)
        }
        if last < 2 { ranges.append(t.structureTokenStart ..< t.structureTokenEnd) }
        if last < 3 { ranges.append(t.keyTokenStart ..< t.keyTokenEnd) }
        if last < 4 { ranges.append(t.fullChordTokenStart ..< t.fullChordTokenEnd) }
        if last <= 5 { ranges.append(t.pitchTokenStart ..< t.pitchTokenEnd) }
        return ranges
    }

    /// Advances on `token`; returns `true` on `<|eos|>`.
    mutating func update(_ token: Int, _ t: SheetSage2Tokenizer) throws -> Bool {
        if token == t.eosToken { return true }
        guard let type = t.tokenType(token) else {
            throw SheetSage2Error.invalidSequence("token \(token) outside the vocabulary")
        }
        if type == .subbeatShift {
            if !inShift && payloadCount > 0 {
                payloadCount = 0
                lastFieldIndex = -1
                incomplete = nil
            }
            inShift = true
            shiftRun += 1
            return false
        }
        inShift = false
        shiftRun = 0
        payloadCount += 1
        switch type {
        case .time: (lastFieldIndex, incomplete) = (0, nil)
        case .meter: (lastFieldIndex, incomplete) = (1, .rhythmAfterMeter)
        case .eighthPosition: (lastFieldIndex, incomplete) = (1, nil)
        case .structure: (lastFieldIndex, incomplete) = (2, nil)
        case .key: (lastFieldIndex, incomplete) = (3, nil)
        case .chordFull: (lastFieldIndex, incomplete) = (4, nil)
        case .pitch: (lastFieldIndex, incomplete) = (5, .melodyAfterPitch)
        case .duration: (lastFieldIndex, incomplete) = (5, nil)
        default: throw SheetSage2Error.invalidSequence("unexpected prompt token type \(type.rawValue)")
        }
        return false
    }
}

/// Greedy, grammar-masked decoding of one window (`constrained_prompt_generate` upstream).
public struct SheetSage2Generator {
    public let model: SheetSage2Model
    public let tokenizer: SheetSage2Tokenizer
    private var maskCache = [PromptGrammarState.MaskKey: MLXArray]()

    public init(model: SheetSage2Model) {
        self.model = model
        tokenizer = SheetSage2Tokenizer(
            audioLengthSeconds: model.config.inputAudioLength, timeHz: model.config.timeHz)
        precondition(tokenizer.nTokens == model.config.vocabSize, "tokenizer and checkpoint vocabularies differ")
    }

    /// Additive mask (`0` allowed, `-inf` forbidden), built once per distinct grammar state.
    private mutating func mask(for key: PromptGrammarState.MaskKey) -> MLXArray {
        if let cached = maskCache[key] { return cached }
        var values = [Float](repeating: -.infinity, count: tokenizer.nTokens)
        for range in PromptGrammarState.allowedRanges(key, tokenizer) {
            for i in range { values[i] = 0 }
        }
        let array = MLXArray(values)
        maskCache[key] = array
        return array
    }

    /// Generates the token sequence (prefix included, `<|eos|>` appended) for one window whose
    /// encoder `memory` is `[1, frames, hidden]`. Stops at `<|eos|>`, at the first time token at
    /// or past `stopTimeSeconds`, or when the sequence reaches `maxSequenceLength`.
    public mutating func generate(
        memory: MLXArray, prompts: [String] = SheetSage2Tokenizer.fullTaskPrompts,
        prefixTokens: [Int]? = nil, stopTimeSeconds: Double?, maxSequenceLength: Int? = nil,
        onToken: ((Int) -> Void)? = nil
    ) throws -> [Int] {
        var prefix = prefixTokens ?? tokenizer.promptPrefix(prompts)
        guard prefix.first == tokenizer.sosToken else {
            throw SheetSage2Error.invalidSequence("generation prefix must begin with <|sos|>")
        }
        if prefix.last == tokenizer.eosToken { prefix.removeLast() }
        guard let outIndex = prefix.firstIndex(of: tokenizer.outToken) else {
            throw SheetSage2Error.invalidSequence("generation prefix is missing <|out|>")
        }
        var state = PromptGrammarState()
        for token in prefix[(outIndex + 1)...] { _ = try state.update(token, tokenizer) }

        let limit = maxSequenceLength ?? model.config.maxOutputSeqLen
        var tokens = prefix
        let cache = model.makeCache()
        var input = MLXArray(prefix.map(Int32.init))[.newAxis]
        while tokens.count < limit {
            let logits = model.decode(input, memory: memory, cache: cache)[0, -1].asType(.float32)
            let allowed = mask(for: state.maskKey)
            let next = argMax(logits + allowed).item(Int.self)
            // NaN logits make argMax return an arbitrary (possibly forbidden) token: fail loudly.
            guard allowed[next].item(Float.self) == 0 else {
                throw SheetSage2Error.invalidSequence("decoder produced non-finite logits at position \(tokens.count)")
            }
            tokens.append(next)
            onToken?(next)
            var finished = try state.update(next, tokenizer)
            if !finished, let stopTimeSeconds, tokenizer.timeTokenStart <= next, next < tokenizer.timeTokenEnd,
                Double(next - tokenizer.timeTokenStart) / Double(tokenizer.timeHz) >= stopTimeSeconds
            {
                tokens.append(tokenizer.eosToken)
                finished = true
            }
            if finished { break }
            input = MLXArray([Int32(next)])[.newAxis]
        }
        if tokens.last != tokenizer.eosToken { tokens.append(tokenizer.eosToken) }
        return tokens
    }
}
