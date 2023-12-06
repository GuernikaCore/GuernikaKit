//
//  ControlNet.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import CoreML
import Accelerate
import Foundation

/// U-Net noise prediction model for stable diffusion
public class ControlNet: ConditioningModule {
    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    let model: ManagedMLModel
    public var configuration: MLModelConfiguration {
        get { model.configuration }
        set { model.configuration = newValue }
    }
    
    public let url: URL
    public let converterVersion: String?
    public let attentionImplementation: AttentionImplementation
    public let method: ConditioningMethod
    public let sampleSize: CGSize
    public let minimumSize: CGSize
    public let maximumSize: CGSize
    public let hiddenSize: Int
    
    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single U-Net  compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public convenience init(modelAt url: URL, configuration: MLModelConfiguration? = nil) throws {
        let metadata = try CoreMLMetadata.metadataForModel(at: url)
        try self.init(modelAt: url, metadata: metadata, configuration: configuration)
    }
    
    public init(modelAt url: URL, metadata: CoreMLMetadata, configuration: MLModelConfiguration? = nil) throws {
        self.url = url
        guard let condInput = metadata.inputSchema[name: "controlnet_cond"] else {
            throw StableDiffusionError.incompatibleControlNet
        }
        let sampleShape = condInput.shape
        sampleSize = CGSize(width: sampleShape[3], height: sampleShape[2])
        if condInput.hasShapeFlexibility {
            minimumSize = CGSize(width: condInput.shapeRange[3][0], height: condInput.shapeRange[2][0])
            maximumSize = CGSize(width: condInput.shapeRange[3][1], height: condInput.shapeRange[2][1])
        } else {
            minimumSize = sampleSize
            maximumSize = sampleSize
        }
        
        hiddenSize = metadata.hiddenSize!
        attentionImplementation = metadata.attentionImplementation
        
        if let info = try? Info.infoForModel(at: url) {
            converterVersion = info.converterVersion
            method = info.method
        } else {
            converterVersion = metadata.userDefinedMetadata?["converter_version"]
            if let methodString = metadata.userDefinedMetadata?["method"],
                let method = ConditioningMethod(rawValue: methodString) {
                self.method = method
            } else {
                self.method = .unknown
            }
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
    ///   - latent: Latent sample
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    /// - Returns: Array of predicted noise residuals
    func predictResiduals(
        input: ConditioningInput,
        latent: MLShapedArray<Float32>,
        timeStep: Double,
        hiddenStates: MLShapedArray<Float32>,
        textEmbeddings: MLShapedArray<Float32>? = nil,
        timeIds: MLShapedArray<Float32>? = nil
    ) throws -> [String: MLShapedArray<Float32>]? {
        guard let image = input.image, input.conditioningScale > 0 else { return nil }
        
        if input.imageData?.shape[2] != latent.shape[2] * 8 || input.imageData?.shape[3] != latent.shape[3] * 8 {
            let inputSize = CGSize(width: latent.shape[3] * 8, height: latent.shape[2] * 8)
            let resizedImage = image.scaledAspectFill(size: inputSize)
            let data = resizedImage.toShapedArray(min: 0.0)
            input.imageData = MLShapedArray<Float32>(concatenating: [data, data], alongAxis: 0)
        }
        guard let imageData = input.imageData else { return nil }
        
        // Batch predict with model
        let result = try model.perform { model in
            let timestepDescription = model.modelDescription.inputDescriptionsByName["timestep"]!
            let timestepShape = timestepDescription.multiArrayConstraint!.shape.map { $0.intValue }
            
            var latent = latent
            var hiddenStates = hiddenStates
            var textEmbeddings = textEmbeddings
            var timeIds = timeIds
            if latent.shape[0] < timestepShape[0] {
                // Make sure the ControlNet input has the expected shape
                latent = MLShapedArray<Float32>(
                    concatenating: Array(repeating: latent, count: timestepShape[0]),
                    alongAxis: 0
                )
                hiddenStates = MLShapedArray<Float32>(
                    concatenating: Array(repeating: hiddenStates, count: timestepShape[0]),
                    alongAxis: 0
                )
                textEmbeddings = textEmbeddings.map { MLShapedArray<Float32>(
                    concatenating: Array(repeating: $0, count: timestepShape[0]),
                    alongAxis: 0
                ) }
                timeIds = timeIds.map { MLShapedArray<Float32>(
                    concatenating: Array(repeating: $0, count: timestepShape[0]),
                    alongAxis: 0
                ) }
            }

            // Match time step batch dimension to the model / latent samples
            let t = MLShapedArray<Float32>(repeating: Float32(timeStep), shape: timestepShape)
            
            var dict: [String: Any] = [
                "sample" : MLMultiArray(latent),
                "timestep" : MLMultiArray(t),
                "encoder_hidden_states": MLMultiArray(hiddenStates),
                "controlnet_cond": MLMultiArray(imageData)
            ]
            if let textEmbeddings, let timeIds {
                dict["text_embeds"] = MLMultiArray(textEmbeddings)
                dict["time_ids"] = MLMultiArray(timeIds)
            }
            let input = try MLDictionaryFeatureProvider(dictionary: dict)
            return try model.prediction(from: input)
        }

        // Pull out the results in Float32 format
        
        // To conform to this func return type make sure we return float32
        // Use the fact that the concatenating constructor for MLMultiArray
        // can do type conversion:
        return result.featureValueDictionary.compactMapValues { value -> MLShapedArray<Float32>? in
            guard let sample = value.multiArrayValue else { return nil }
            var noise = MLShapedArray<Float32>(MLMultiArray(
                concatenating: [sample],
                axis: 0,
                dataType: .float32
            ))
            if noise.shape[0] > latent.shape[0] {
                noise = MLShapedArray(
                    scalars: noise[1].scalars,
                    shape: noise.shape.replacing([noise.shape[0]], with: [latent.shape[0]], maxReplacements: 1)
                )
            }
            noise.scale(input.conditioningScale)
            return noise
        }
    }
}

extension Array where Element == ConditioningInput {
    var controlNets: [ControlNet] {
        compactMap { input -> ControlNet? in
            input.module as? ControlNet
        }
    }
    
    func predictControlNetResiduals(
        latent: MLShapedArray<Float32>,
        timeStep: Double,
        hiddenStates: MLShapedArray<Float32>,
        textEmbeddings: MLShapedArray<Float32>? = nil,
        timeIds: MLShapedArray<Float32>? = nil,
        reduceMemory: Bool
    ) throws -> [String: MLShapedArray<Float32>]? {
        try compactMap { input -> [String: MLShapedArray<Float32>]? in
            guard let controlNet = input.module as? ControlNet else { return nil }
            let residuals = try controlNet.predictResiduals(
                input: input,
                latent: latent,
                timeStep: timeStep,
                hiddenStates: hiddenStates,
                textEmbeddings: textEmbeddings, 
                timeIds: timeIds
            )
            if reduceMemory {
                controlNet.unloadResources()
            }
            return residuals
        }.addResiduals()
    }
}

extension ControlNet {
    struct Info: Decodable {
        let id: String
        let converterVersion: String?
        let method: ConditioningMethod
        
        enum CodingKeys: String, CodingKey {
            case id = "identifier"
            case converterVersion = "converter_version"
            case method
        }
        
        static func infoForModel(at url: URL) throws -> Info? {
            let moduleUrl = url.appendingPathComponent("guernika.json")
            if FileManager.default.fileExists(atPath: moduleUrl.path(percentEncoded: false)) {
                return try JSONDecoder().decode(Info.self, from: Data(contentsOf: moduleUrl))
            }
            return nil
        }
    }
}
