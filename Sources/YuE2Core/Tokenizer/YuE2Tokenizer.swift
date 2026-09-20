// YuE2Tokenizer.swift - Qwen2.5-derived BPE tokenizer, no special-token recognition (plan §2.7)
// Copyright 2026 Vincent Gourbin

import Foundation
import Tokenizers

/// Encodes exactly like the checkpoint-native `qwen.tiktoken` (NFC, no special tokens
/// recognized in free text). Backed by `$modelDir/tokenizer.json`, converted from
/// Qwen2.5-0.5B by `Scripts/convert-tokenizer.py` (T-1.5): same vocabulary and merges as
/// upstream, `added_tokens` emptied so strings like `<|im_start|>` fall through to
/// ordinary BPE instead of being recognized as special tokens.
public final class YuE2Tokenizer: TextDecoding, @unchecked Sendable {
    private let tokenizer: Tokenizer

    private init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Loads `$modelDir/tokenizer.json` (+ `tokenizer_config.json`) via swift-transformers.
    public static func load(modelDir: URL) async throws -> YuE2Tokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        let instance = YuE2Tokenizer(tokenizer: tokenizer)
        // `added_tokens` must have been emptied by the conversion script — if it hadn't,
        // "<|endoftext|>" would encode to the single reserved id 151643 instead of being
        // split into ordinary BPE pieces (plan §2.7, piège n°16 territory: a stale/hand-
        // edited tokenizer.json would silently reintroduce special-token recognition).
        precondition(
            !instance.encode("<|endoftext|>").contains(YuE2Token.eod),
            "tokenizer.json still recognizes <|endoftext|> as a special token — added_tokens was not emptied"
        )
        return instance
    }

    /// NFC-normalizes, then encodes without recognizing any special token.
    public func encode(_ text: String) -> [Int] {
        let normalized = text.precomposedStringWithCanonicalMapping
        return tokenizer.encode(text: normalized, addSpecialTokens: false)
    }

    /// Decodes, silently dropping any id outside the ordinary text vocabulary (`< EOD`).
    public func decode(_ ids: [Int]) -> String {
        let ordinary = ids.filter { $0 >= 0 && $0 < YuE2Token.eod }
        return tokenizer.decode(tokens: ordinary, skipSpecialTokens: false)
    }
}
