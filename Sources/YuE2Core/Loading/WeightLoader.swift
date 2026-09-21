// WeightLoader.swift - generic strict safetensors loader (plan §2.5, pieges n°8/9)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Loads a `.safetensors` file and applies its tensors to a module, refusing any mismatch
/// between what the checkpoint provides and what the module's parameter tree expects.
public enum WeightLoader {
    /// Loads every tensor in `url`, remapping/dropping keys through `filter` (`nil` drops).
    public static func load(url: URL, filter: (String) -> String?) throws -> [String: MLXArray] {
        let raw = try loadArrays(url: url)
        var result = [String: MLXArray]()
        result.reserveCapacity(raw.count)
        for (key, value) in raw {
            guard let mapped = filter(key) else { continue }
            result[mapped] = value
        }
        return result
    }

    /// Applies `weights` to `module`, throwing if any expected parameter has no tensor or any
    /// tensor matches no parameter — a silent partial load is worse than a crash here.
    ///
    /// `retaining` (stage-scoped residency, `WeightResidency.swift`): when given, the coverage
    /// check still runs over every key, but only the tensors it selects are applied and
    /// evaluated — the others stay as the module's own unevaluated init zeros, i.e. cost no
    /// memory until `loadWeights(of:)` brings them in. `loadArrays` is lazy (safetensors are
    /// memory-mapped and each tensor materializes on its own `eval`), so skipping the eval is
    /// what makes the skipped branch free.
    public static func apply(
        _ weights: [String: MLXArray], to module: Module, component: String,
        retaining: ((String) -> Bool)? = nil
    ) throws {
        let expected = Set(module.parameters().flattened().map(\.0))
        let provided = Set(weights.keys)
        let missing = expected.subtracting(provided)
        let unexpected = provided.subtracting(expected)
        guard missing.isEmpty, unexpected.isEmpty else {
            let missingList = missing.sorted().prefix(10).joined(separator: ", ")
            let unexpectedList = unexpected.sorted().prefix(10).joined(separator: ", ")
            throw YuE2Error.weightMismatch(
                "\(component): missing=[\(missingList)] unexpected=[\(unexpectedList)]"
            )
        }
        let retained = retaining.map { keep in weights.filter { keep($0.key) } } ?? weights
        module.update(parameters: ModuleParameters.unflattened(retained))
        // Piège n°8: eval tensor by tensor, never `eval(model.parameters())` on the whole tree
        // at once — that materializes everything simultaneously and can OOM silently.
        for (_, value) in retained {
            eval(value)
        }
    }
}
