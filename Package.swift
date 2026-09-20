// swift-tools-version: 6.0
// Package.swift - YuE2 music generation for Apple Silicon (MLX)
// Copyright 2026 Vincent Gourbin

import PackageDescription

let package = Package(
    name: "YuE2Swift",
    platforms: [.macOS(.v15)],
    products: [
        // MARK: - Libraries
        .library(name: "YuE2Core", targets: ["YuE2Core"]),
        // MARK: - CLI Tools
        .executable(name: "yue2", targets: ["YuE2CLI"]),
        .executable(name: "yue2-bench-ui", targets: ["YuE2BenchUI"]),
    ],
    dependencies: [
        // Pinned exact: mlx-swift has broken API in patch releases before; bump deliberately.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),  // Tokenizers only
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.5.0"),
    ],
    targets: [
        // MARK: - Libraries
        .target(
            name: "YuE2Core",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        // MARK: - CLI Tools
        .executableTarget(
            name: "YuE2CLI",
            dependencies: [
                "YuE2Core",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        .executableTarget(
            name: "YuE2BenchUI",
            dependencies: [
                "YuE2Core",
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        // MARK: - Tests
        .testTarget(
            name: "YuE2Tests",
            dependencies: ["YuE2Core"]
        ),
    ]
)
