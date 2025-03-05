// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "SimpleMetalApp",
    platforms: [
        .macOS(.v12)  // Minimum macOS version for modern Metal support
    ],
    products: [
        .executable(
            name: "SimpleMetalApp",
            targets: ["SimpleMetalApp"]),
    ],
    dependencies: [
        // No external dependencies needed as Metal is part of the system frameworks
    ],
    targets: [
        .executableTarget(
            name: "SimpleMetalApp",
            dependencies: [],
            path: "src",
            sources: ["simple.swift"]  // Explicitly include only the Swift file
        )
    ]
)
