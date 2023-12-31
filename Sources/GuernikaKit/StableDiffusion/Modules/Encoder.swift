//
//  Encoder.swift
//  
//
//  Created by Guillermo Cique Fernández on 24/5/23.
//

import CoreML
import Accelerate
import Foundation
import RandomGenerator

/// A encoder model which produces RGB images from latent samples
public class Encoder {
    /// VAE encoder model
    let model: ManagedMLModel
    public var configuration: MLModelConfiguration {
        get { model.configuration }
        set { model.configuration = newValue }
    }
    
    public let sampleSize: CGSize
    let cache: Cache<CGImage, MLShapedArray<Float32>> = Cache(maxItems: 2)
    let scaleFactor: Float32?
    
    /// Create encoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE encoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A encoder that will lazily load its required resources when needed or requested
    public convenience init(modelAt url: URL, configuration: MLModelConfiguration) throws {
        let metadata = try CoreMLMetadata.metadataForModel(at: url)
        try self.init(modelAt: url, metadata: metadata, configuration: configuration)
    }
    
    public init(modelAt url: URL, metadata: CoreMLMetadata, configuration: MLModelConfiguration) throws {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
        
        scaleFactor = metadata.userDefinedMetadata?["scaling_factor"].flatMap { Float32($0) }
        let inputImageShape = metadata.inputSchema[name: "z"]!.shape
        let width: Int = inputImageShape[3]
        let height: Int = inputImageShape[2]
        sampleSize = CGSize(width: width, height: height)
    }
    
    /// Unload the underlying model to free up memory
    public func unloadResources(clearCache: Bool = false) {
        model.unloadResources()
        if clearCache {
            cache.clear()
        }
    }
    
    /// Encode image into latent samples
    ///
    ///  - Parameters:
    ///    - image: Input image
    ///  - Returns: Latent samples to decode
    public func encode(
        _ image: CGImage,
        scaleFactor: Float32 = 0.18215,
        generator: RandomGenerator
    ) throws -> MLShapedArray<Float32> {
        if let cachedLatent = cache[image] {
            // Run the generator to simulate encoder using the generator
            let _ = generator.nextArray(shape: cachedLatent.shape)
            return cachedLatent
        }
        let resizedImage = image.scaledAspectFill(size: sampleSize)
        let imageData = resizedImage.toShapedArray()
        let latent = try encode(imageData, scaleFactor: scaleFactor, generator: generator)
        cache[image] = latent
        return latent
    }
    
    public func encode(
        _ imageData: MLShapedArray<Float32>,
        scaleFactor: Float32 = 0.18215,
        generator: RandomGenerator
    ) throws -> MLShapedArray<Float32> {
        
        let result = try model.perform { model in
            let inputName = model.modelDescription.inputDescriptionsByName.first!.key
            let dict = [inputName: MLMultiArray(imageData)]
            let input = try MLDictionaryFeatureProvider(dictionary: dict)
            return try model.prediction(from: input)
        }
        let outputName = result.featureNames.first!
        let outputValue = result.featureValue(for: outputName)!.multiArrayValue!
        let output = MLShapedArray<Float32>(converting: outputValue)
        
        let latent: MLShapedArray<Float32>
        let latentShape: [Int]
        
        if output.shape[1] > 4 {
            // DiagonalGaussianDistribution
            let mean = output[0][0..<4]
            let std = MLShapedArray<Float32>(
                scalars: output[0][4..<8].scalars.map {
                    let logvar = min(max($0, -30), 20)
                    return exp(0.5 * logvar)
                },
                shape: mean.shape
            )
            latent = MLShapedArray<Float32>(
                converting: generator.nextArray(
                    shape: mean.shape,
                    mean: mean.scalars.map { Double($0) },
                    stdev: std.scalars.map { Double($0) }
                )
            )
            latentShape = [1] + latent.shape
        } else {
            // TinyEncoder does not need DiagonalGaussianDistribution
            latent = output
            latentShape = output.shape
        }
        
        // Reference pipeline scales the latent after encoding
        let scaleFactor = self.scaleFactor ?? scaleFactor
        guard scaleFactor != 1 else { return latent }
        let latentScaled = MLShapedArray(unsafeUninitializedShape: latentShape) { scalars, _ in
            latent.withUnsafeShapedBufferPointer { latent, _, _ in
                vDSP.multiply(scaleFactor, latent, result: &scalars)
            }
        }
        
        return latentScaled
    }
}
