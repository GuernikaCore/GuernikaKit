// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "GuernikaKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "GuernikaKit", targets: ["GuernikaKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/GuernikaCore/RandomGenerator.git", from: "1.0.0"),
        .package(url: "https://github.com/GuernikaCore/Schedulers.git", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "GuernikaKit",
            dependencies: [
                .product(name: "RandomGenerator", package: "RandomGenerator"),
                .product(name: "Schedulers", package: "Schedulers")
            ]
        ),
        .testTarget(
            name: "GuernikaKitTests",
            dependencies: ["GuernikaKit"]
        ),
    ]
)
