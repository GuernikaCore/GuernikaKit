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
public class ControlNet {
    public enum Method: String, Hashable, Decodable {
        case canny
        case depth
        case pose
        case mlsd
        case normal
        case scribble
        case hed
        case segmentation
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
    
    public class Input: Identifiable {
        public let id: UUID = UUID()
        public let controlNet: ControlNet
        public var method: Method { controlNet.method }
        public var conditioningScale: Float = 1.0
        public var image: CGImage? {
            didSet {
                // Reset imageData, it will be updated when run with the correct size
                imageData = nil
            }
        }
        var imageData: MLShapedArray<Float32>?
        
        public init(controlNet: ControlNet) {
            self.controlNet = controlNet
        }
    }
    
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
    public let method: Method
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
    public init(modelAt url: URL, configuration: MLModelConfiguration? = nil) throws {
        self.url = url
        let metadata = try CoreMLMetadata.metadataForModel(at: url)
        let condInput = metadata.inputSchema[name: "controlnet_cond"]!
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
            if let methodString = metadata.userDefinedMetadata?["method"], let method = Method(rawValue: methodString) {
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
        input: Input,
        latent: MLShapedArray<Float32>,
        timeStep: Double,
        hiddenStates: MLShapedArray<Float32>,
        textEmbeddings: MLShapedArray<Float32>? = nil,
        timeIds: MLShapedArray<Float32>? = nil
    ) throws -> ([MLShapedArray<Float32>], MLShapedArray<Float32>)? {
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

        let downResNoise = [
            result.featureValue(for: "down_block_res_samples_00")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_01")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_02")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_03")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_04")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_05")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_06")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_07")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_08")!.multiArrayValue!,
            result.featureValue(for: "down_block_res_samples_09")?.multiArrayValue,
            result.featureValue(for: "down_block_res_samples_10")?.multiArrayValue,
            result.featureValue(for: "down_block_res_samples_11")?.multiArrayValue,
        ]
        let downResFp32Noise: [MLShapedArray<Float32>] = downResNoise.compactMap { sample in
            guard let sample else { return nil }
            var noise = MLShapedArray<Float32>(MLMultiArray(
                concatenating: [sample],
                axis: 0,
                dataType: .float32
            ))
            noise.scale(input.conditioningScale)
            return noise
        }

        let midResNoise = result.featureValue(for: "mid_block_res_sample")!.multiArrayValue!
        var midResFp32Noise = MLShapedArray<Float32>(MLMultiArray(
            concatenating: [midResNoise],
            axis: 0,
            dataType: .float32
        ))
        midResFp32Noise.scale(input.conditioningScale)
        return (downResFp32Noise, midResFp32Noise)
    }
}

extension Array where Element == ControlNet.Input {
    func predictResiduals(
        latent: MLShapedArray<Float32>,
        timeStep: Double,
        hiddenStates: MLShapedArray<Float32>,
        textEmbeddings: MLShapedArray<Float32>? = nil,
        timeIds: MLShapedArray<Float32>? = nil
    ) throws -> ([MLShapedArray<Float32>], MLShapedArray<Float32>)? {
        try compactMap { input in
            try input.controlNet.predictResiduals(
                input: input,
                latent: latent,
                timeStep: timeStep,
                hiddenStates: hiddenStates,
                textEmbeddings: textEmbeddings, 
                timeIds: timeIds
            )
        }.addResiduals()
    }
}

fileprivate extension Array where Element == ([MLShapedArray<Float32>], MLShapedArray<Float32>) {
    func addResiduals() -> ([MLShapedArray<Float32>], MLShapedArray<Float32>)? {
        guard count > 1 else { return first }
        var total: ([MLShapedArray<Float32>], MLShapedArray<Float32>) = self[0]
        for res in self.dropFirst() {
            let stride = vDSP_Stride(1)
            // Go through each latent residuals of the total and current ControlNet residuals
            for downSample in 0..<total.0.count {
                total.0[downSample].withUnsafeMutableShapedBufferPointer { total, _, _ in
                    res.0[downSample].withUnsafeShapedBufferPointer { latent, _, _ in
                        let n = vDSP_Length(total.count)
                        vDSP_vadd(
                            total.baseAddress!, stride,
                            latent.baseAddress!, stride,
                            total.baseAddress!, stride,
                            n
                        )
                    }
                }
            }
            
            total.1.withUnsafeMutableShapedBufferPointer { total, _, _ in
                res.1.withUnsafeShapedBufferPointer { latent, _, _ in
                    let n = vDSP_Length(total.count)
                    vDSP_vadd(
                        total.baseAddress!, stride,
                        latent.baseAddress!, stride,
                        total.baseAddress!, stride,
                        n
                    )
                }
            }
        }
        return total
    }
}

extension ControlNet {
    struct Info: Decodable {
        let id: String
        let converterVersion: String?
        let method: Method
        
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
