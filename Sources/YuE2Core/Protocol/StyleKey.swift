// StyleKey.swift - make the planned score follow the key and tempo the style asks for (ASK Q12)
// Copyright 2026 Vincent Gourbin

import Foundation

/// The style line's "key of D major" / "A minor" / "120 BPM" is text conditioning, not a
/// constraint: the planner samples its own key and tempo (in the app's sample, 1 in 7 songs kept
/// the style's key). A forced score header (`SongRequest.abcPrefix`) makes them binding; this
/// builds it from the style.
public enum StyleKey {
    /// Key (as an ABC `K:` value: `D`, `F#m`, `Bb`) and tempo (BPM) stated in `style`, if any.
    public static func parse(_ style: String) -> (key: String?, bpm: Int?) {
        (key(in: style), bpm(in: style))
    }

    static func key(in style: String) -> String? {
        let text = style.replacingOccurrences(of: "♯", with: "#").replacingOccurrences(of: "♭", with: "b")
        let patterns = [
            #"(?i)\b([A-G])([#b]?)\s*(major|minor|maj|min)\b"#,  // "D major", "F# minor"
            #"(?i)\bkey of ([A-G])([#b]?)(?![a-z])"#,             // "key of C" (major)
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern),
                let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
            else { continue }
            let group = { (i: Int) -> String in
                match.range(at: i).location == NSNotFound ? "" : String(text[Range(match.range(at: i), in: text)!])
            }
            let mode = match.numberOfRanges > 3 ? group(3).lowercased() : "major"
            return abcKey(letter: group(1).uppercased(), accidental: group(2).lowercased(), minor: mode.hasPrefix("min"))
        }
        return nil
    }

    static func bpm(in style: String) -> Int? {
        guard let regex = try? NSRegularExpression(pattern: #"(?i)\b(\d{2,3})\s*bpm\b"#),
            let match = regex.firstMatch(in: style, range: NSRange(style.startIndex..., in: style)),
            let value = Int(style[Range(match.range(at: 1), in: style)!]), (30...300).contains(value)
        else { return nil }
        return value
    }

    /// A key signature the model writes: the spelling given when it is a standard signature,
    /// else its enharmonic one (A# major → Bb, D# minor → Ebm).
    static func abcKey(letter: String, accidental: String, minor: Bool) -> String {
        let pc = (["C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11][letter]! + (accidental == "#" ? 1 : accidental == "b" ? -1 : 0) + 12) % 12
        let majors = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        let minors = ["Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm"]
        let valid: Set<String> = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb",
                                  "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm", "Abm"]
        let spelled = letter + accidental + (minor ? "m" : "")
        return valid.contains(spelled) ? spelled : (minor ? minors[pc] : majors[pc])
    }

    /// The opening of a score in the planner's own dialect, ending on `K:`: the planner writes the
    /// sections after it. `nil` when the style states neither a key nor a tempo.
    public static func scoreHeader(style: String) -> String? {
        let (key, bpm) = parse(style)
        guard key != nil || bpm != nil else { return nil }
        var lines = ["X:1", "T:", "M:4/4", "L:1/16"]
        if let bpm { lines.append("Q:1/4=\(bpm)") }
        lines += [
            "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
            "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"",
        ]
        if let key { lines.append("K:\(key)") }
        return lines.joined(separator: "\n") + "\n"
    }
}

extension SongRequest {
    /// The same request with a forced score header taken from its style (key and/or tempo), so
    /// the planner keeps them; unchanged when a score or a prefix is already imposed, or when the
    /// style states neither.
    public func followingStyleKey() -> SongRequest {
        guard abc == nil, abcPrefix == nil, let header = StyleKey.scoreHeader(style: style) else { return self }
        var copy = self
        copy.abcPrefix = header
        return copy
    }
}
