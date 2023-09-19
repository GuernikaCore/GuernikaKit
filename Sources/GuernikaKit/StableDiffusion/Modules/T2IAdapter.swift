//
//  T2IAdapter.swift
//
//
//  Created by Guillermo Cique Fernández on 10/9/23.
//

import CoreML
import Accelerate
import Foundation

/// A simple ResNet-like model that accepts images containing control signals such as keyposes and depth
public class T2IAdapter: ConditioningModule {
    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    let model: ManagedMLModel
    var configuration: MLModelConfiguration {
        get { model.configuration }
        set { model.configuration = newValue }
    }
    
    public let url: URL
    public let converterVersion: String?
    public let attentionImplementation: AttentionImplementation
    public let method: ConditioningMethod
    let inChannels: Int
    public let sampleSize: CGSize
    public let minimumSize: CGSize
    public let maximumSize: CGSize
    
    /// Creates a T2IAdapter model
    ///
    /// - Parameters:
    ///   - url: Location of single U-Net  compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration? = nil) throws {
        self.url = url
        let metadata = try CoreMLMetadata.metadataForModel(at: url)
        guard let input = metadata.inputSchema[name: "input"] else {
            throw StableDiffusionError.incompatibleAdapter
        }
        let inputShape = input.shape
        inChannels = inputShape[1]
        sampleSize = CGSize(width: inputShape[3], height: inputShape[2])
        if input.hasShapeFlexibility {
            minimumSize = CGSize(width: input.shapeRange[3][0], height: input.shapeRange[2][0])
            maximumSize = CGSize(width: input.shapeRange[3][1], height: input.shapeRange[2][1])
        } else {
            minimumSize = sampleSize
            maximumSize = sampleSize
        }
        
        attentionImplementation = metadata.attentionImplementation
        
        converterVersion = metadata.userDefinedMetadata?["converter_version"]
        if let methodString = metadata.userDefinedMetadata?["method"],
           let method = ConditioningMethod(rawValue: methodString) {
            self.method = method
        } else {
            self.method = .unknown
        }
        
        let configuration = configuration ?? attentionImplementation.preferredModelConfiguration
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }
    
    /// Unload the underlying model to free up memory
    public func unloadResources() {
        model.unloadResources()
    }
    
    /// Batch prediction noise from latent samples
    ///
    /// - Parameters:
    ///   - input: Input data including image and conditioning
    ///   - latent: Latent sample
    /// - Returns: Array of predicted noise residuals
    func predictResiduals(
        input: ConditioningInput,
        latent: MLShapedArray<Float32>
    ) throws -> [String: MLShapedArray<Float32>]? {
        guard let image = input.image, input.conditioningScale > 0 else { return nil }
        
        if input.imageData?.shape[2] != latent.shape[2] * 8 || input.imageData?.shape[3] != latent.shape[3] * 8 {
            let inputSize = CGSize(width: latent.shape[3] * 8, height: latent.shape[2] * 8)
            let resizedImage = image.scaledAspectFill(size: inputSize)
            if inChannels == 1 {
                // Black and white input expected
                input.imageData = resizedImage.toShapedArray(min: 0.0, components: [false, false, false, true])
            } else {
                input.imageData = resizedImage.toShapedArray(min: 0.0)
            }
        }
        guard let imageData = input.imageData else { return nil }
        
        // Batch predict with model
        let result = try model.perform { model in
            var dict: [String: Any] = [
                "input": MLMultiArray(imageData)
            ]         
            let input = try MLDictionaryFeatureProvider(dictionary: dict)
            return try model.prediction(from: input)
        }

        // Pull out the results in Float32 format
        
        // To conform to this func return type make sure we return float32
        // Use the fact that the concatenating constructor for MLMultiArray
        // can do type conversion:

        return result.featureValueDictionary.compactMapValues { value in
            guard let sample = value.multiArrayValue else { return nil }
            var noise = MLShapedArray<Float32>(MLMultiArray(
                concatenating: [sample, sample],
                axis: 0,
                dataType: .float32
            ))
            noise.scale(input.conditioningScale)
            return noise
        }
    }
}

extension Array where Element == ConditioningInput {
    var adapters: [T2IAdapter] {
        compactMap { input -> T2IAdapter? in
            input.module as? T2IAdapter
        }
    }
    
    func predictAdapterResiduals(latent: MLShapedArray<Float32>) throws -> [String: MLShapedArray<Float32>]? {
        try compactMap { input -> [String: MLShapedArray<Float32>]? in
            guard let adapter = input.module as? T2IAdapter else { return nil }
            return try adapter.predictResiduals(
                input: input,
                latent: latent
            )
        }.addResiduals()
    }
}
