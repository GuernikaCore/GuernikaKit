//
//  Unet.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import CoreML
import Schedulers
import Foundation

/// U-Net noise prediction model for stable diffusion
public class Unet {
    public enum Function: String, Decodable {
        case standard
        case refiner
        case inpaint
        case instructions
        case unknown
        
        public init(from decoder: Swift.Decoder) throws {
            do {
                let container = try decoder.singleValueContainer()
                let rawValue = try container.decode(String.self)
                if let value = Self(rawValue: rawValue) {
                    self = value
                } else {
                    self = .unknown
                }
            } catch {
                self = .unknown
            }
        }
    }

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    let models: [ManagedMLModel]
    var reduceMemory: Bool = false
    public var configuration: MLModelConfiguration {
        get { models[0].configuration }
        set {
            for model in models {
                model.configuration = newValue
            }
        }
    }
    
    public let converterVersion: String?
    public let attentionImplementation: AttentionImplementation
    public let function: Function
    public let predictionType: PredictionType
    public let schedulerSetAlphaToOne: Bool
    public let schedulerStepsOffset: Int
    public let timestepSpacing: TimestepSpacing?
    
    /// The expected shape of the models timestemp input
    let timestepShape: [Int]
    let latentSampleShape: [Int]
    public var batchSize: Int { latentSampleShape[0] }
    public let sampleSize: CGSize
    public let minimumSize: CGSize
    public let maximumSize: CGSize
    public let supportsAdapter: Bool
    public let supportsControlNet: Bool
    public let hiddenSize: Int
    public let timeCondProjDim: Int?

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single U-Net  compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public convenience init(modelAt url: URL, configuration: MLModelConfiguration? = nil) throws {
        try self.init(chunksAt: [url], configuration: configuration)
    }

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - urls: Location of chunked U-Net via urls to each compiled chunk
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public convenience init(chunksAt urls: [URL], configuration: MLModelConfiguration? = nil) throws {
        let metadata = try CoreMLMetadata.metadataForModel(at: urls[0])
        self.init(chunksAt: urls, metadata: metadata, configuration: configuration)
    }
    
    public init(chunksAt urls: [URL], metadata: CoreMLMetadata, configuration: MLModelConfiguration? = nil) {
        timestepShape = metadata.inputSchema[name: "timestep"]!.shape
        let sampleInput = metadata.inputSchema[name: "sample"]!
        let sampleShape = sampleInput.shape
        latentSampleShape = sampleShape
        sampleSize = CGSize(width: sampleShape[3] * 8, height: sampleShape[2] * 8)
        if sampleInput.hasShapeFlexibility {
            minimumSize = CGSize(width: sampleInput.shapeRange[3][0] * 8, height: sampleInput.shapeRange[2][0] * 8)
            maximumSize = CGSize(width: sampleInput.shapeRange[3][1] * 8, height: sampleInput.shapeRange[2][1] * 8)
        } else {
            minimumSize = sampleSize
            maximumSize = sampleSize
        }
        supportsAdapter = metadata.inputSchema[name: "adapter_res_samples_00"] != nil
        supportsControlNet = metadata.inputSchema[name: "mid_block_res_sample"] != nil
        hiddenSize = metadata.hiddenSize!
        timeCondProjDim = metadata.inputSchema[name: "timestep_cond"]?.shape[1]
        attentionImplementation = metadata.attentionImplementation
        
        if let info = try? Info.infoForModel(at: urls[0]) {
            converterVersion = info.converterVersion
            function = info.function
            predictionType = info.predictionType ?? .epsilon
        } else {
            converterVersion = metadata.userDefinedMetadata?["converter_version"]
            if metadata.userDefinedMetadata?["requires_aesthetics_score"] == "true" {
                function = .refiner
            } else if timestepShape[0] == 3 {
                function = .instructions
            } else if sampleShape[1] == 9 {
                function = .inpaint
            } else {
                function = .standard
            }
            if let predictionTypeString = metadata.userDefinedMetadata?["prediction_type"] {
                predictionType = PredictionType(rawValue: predictionTypeString) ?? .epsilon
            } else {
                predictionType = .epsilon
            }
        }
        if let timestepSpacingString = metadata.userDefinedMetadata?["timestep_spacing"] {
            timestepSpacing = TimestepSpacing(rawValue: timestepSpacingString) ?? .leading
        } else {
            timestepSpacing = .leading
        }
        if let stepsOffsetString = metadata.userDefinedMetadata?["steps_offset"], let stepsOffset = Int(stepsOffsetString) {
            schedulerStepsOffset = stepsOffset
        } else {
            schedulerStepsOffset = 1
        }
        if let setAlphaToOne = metadata.userDefinedMetadata?["set_alpha_to_one"] {
            schedulerSetAlphaToOne = setAlphaToOne == "true"
        } else {
            schedulerSetAlphaToOne = false
        }
        
        let configuration = configuration ?? attentionImplementation.preferredModelConfiguration
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }

    /// Batch prediction noise from latent samples
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    /// - Returns: Array of predicted noise residuals
    func predictNoise(
        latents: [MLShapedArray<Float32>],
        additionalResiduals: [String: MLShapedArray<Float32>]? = nil,
        timeStep: Double,
        timeStepCond: MLShapedArray<Float32>? = nil,
        hiddenStates: MLShapedArray<Float32>,
        textEmbeddings: MLShapedArray<Float32>? = nil,
        timeIds: MLShapedArray<Float32>? = nil
    ) throws -> [MLShapedArray<Float32>] {
        // Match time step batch dimension to the model / latent samples
        let t = MLShapedArray<Float32>(repeating: Float32(timeStep), shape: timestepShape)

        // Form batch input to model
        let inputs: [MLDictionaryFeatureProvider] = try latents.map { latent in
            var dict: [String: Any] = [
                "sample" : MLMultiArray(latent),
                "timestep" : MLMultiArray(t),
                "encoder_hidden_states": MLMultiArray(hiddenStates)
            ]
            if let timeStepCond {
                dict["timestep_cond"] = MLMultiArray(timeStepCond)
            }
            if let textEmbeddings, let timeIds {
                dict["text_embeds"] = MLMultiArray(textEmbeddings)
                dict["time_ids"] = MLMultiArray(timeIds)
            }
            
            if let additionalResiduals {
                for (name, residual) in additionalResiduals {
                    dict[name] = MLMultiArray(residual)
                }
            }
            
            try models.first?.perform { model in
                let inputDescriptions = model.modelDescription.inputDescriptionsByName
                for input in inputDescriptions {
                    guard dict[input.key] == nil else { continue }
                    dict[input.key] = input.value.residualEmptyValue(
                        latentShape: latent.shape,
                        defaultShape: latentSampleShape
                    )
                }
            }
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Make predictions
        let results = try predictions(from: batch)

        // Pull out the results in Float32 format
        let noise = (0..<results.count).map { i in

            let result = results.features(at: i)
            let outputName = result.featureNames.first!

            let outputNoise = result.featureValue(for: outputName)!.multiArrayValue!

            // To conform to this func return type make sure we return float32
            // Use the fact that the concatenating constructor for MLMultiArray
            // can do type conversion:
            let fp32Noise = MLMultiArray(
                concatenating: [outputNoise],
                axis: 0,
                dataType: .float32
            )
            return MLShapedArray<Float32>(fp32Noise)
        }

        return noise
    }

    private func predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {
        var results = try models[0].perform { model in
            try model.predictions(fromBatch: batch)
        }

        if models.count == 1 {
            return results
        }
        
        if reduceMemory {
            models.first?.unloadResources()
        }
        
        // Manual pipeline batch prediction
        let inputs = batch.arrayOfFeatureValueDictionaries
        for stage in models.dropFirst() {
            // Combine the original inputs with the outputs of the last stage
            let next = try results.arrayOfFeatureValueDictionaries
                .enumerated()
                .map { index, dict in
                    let nextDict =  dict.merging(inputs[index]) { out, _ in out }
                    return try MLDictionaryFeatureProvider(dictionary: nextDict)
                }
            let nextBatch = MLArrayBatchProvider(array: next)

            // Predict
            results = try stage.perform { model in
                try model.predictions(fromBatch: nextBatch)
            }
            if reduceMemory {
                stage.unloadResources()
            }
        }

        return results
    }
}

extension MLFeatureProvider {
    var featureValueDictionary: [String : MLFeatureValue] {
        self.featureNames.reduce(into: [String : MLFeatureValue]()) { result, name in
            result[name] = self.featureValue(for: name)
        }
    }
}

extension MLFeatureDescription {
    var emptyValue: MLMultiArray {
        MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: multiArrayConstraint!.shape.map { $0.intValue }))
    }
    
    fileprivate func residualEmptyValue(latentShape: [Int], defaultShape: [Int]) -> MLMultiArray {
        var valueShape = multiArrayConstraint!.shape.map { $0.intValue }
        // This makes sure the inputs of empty residuals have the correct size when the latent shape is not the default
        if latentShape != defaultShape {
            valueShape[2] = Int(Double(valueShape[2]) / Double(defaultShape[2]) * Double(latentShape[2]))
            valueShape[3] = Int(Double(valueShape[3]) / Double(defaultShape[3]) * Double(latentShape[3]))
        }
        return MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: valueShape))
    }
}

extension MLBatchProvider {
    var arrayOfFeatureValueDictionaries: [[String : MLFeatureValue]] {
        (0..<self.count).map {
            self.features(at: $0).featureValueDictionary
        }
    }
}

extension Unet {
    struct Info: Decodable {
        let id: String
        let converterVersion: String?
        let function: Function
        let predictionType: PredictionType?
        
        enum CodingKeys: String, CodingKey {
            case id = "identifier"
            case converterVersion = "converter_version"
            case function
            case predictionType = "prediction_type"
        }
        
        static func infoForModel(at url: URL) throws -> Info? {
            let moduleUrl = url.appendingPathComponent("guernika.json")
            if FileManager.default.fileExists(atPath: moduleUrl.absoluteURL.path(percentEncoded: false)) {
                return try JSONDecoder().decode(Info.self, from: Data(contentsOf: moduleUrl))
            }
            let pipelineUrl = url.deletingLastPathComponent().appendingPathComponent("guernika.json")
            if FileManager.default.fileExists(atPath: pipelineUrl.absoluteURL.path(percentEncoded: false)) {
                return try JSONDecoder().decode(Info.self, from: Data(contentsOf: pipelineUrl))
            }
            return nil
        }
    }
}
