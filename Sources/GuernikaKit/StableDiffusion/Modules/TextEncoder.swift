//
//  TextEncoder.swift
//  
//
//  Created by Guillermo Cique Fernández on 23/5/23.
//

import CoreML
import Foundation

///  A model for encoding text
public class TextEncoder {
    /// Text tokenizer
    public internal(set) lazy var tokenizer: BPETokenizer = {
        try! BPETokenizer(
            mergesUrl: baseUrl.appending(component: "merges.txt"),
            vocabularyUrl: baseUrl.appending(component: "vocab.json"),
            addedVocabUrl: modelUrl.appending(component: "added_vocab.json")
        )
    }()
    let baseUrl: URL
    let modelUrl: URL

    /// Embedding model
    let model: ManagedMLModel
    var configuration: MLModelConfiguration {
        get { model.configuration }
        set { model.configuration = newValue }
    }
    
    public let maxInputLength: Int
    public let hiddenSize: Int
    let cache: Cache<String, (MLShapedArray<Float32>, MLShapedArray<Float32>)> = Cache(maxItems: 4)

    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - url: Location of compiled text encoding  Core ML model
    ///   - configuration: configuration to be used when the model is loaded
    /// - Returns: A text encoder that will lazily load its required resources when needed or requested
    public init(
        modelAt url: URL,
        baseUrl: URL? = nil,
        padToken: String? = nil,
        configuration: MLModelConfiguration? = nil
    ) throws {
        self.modelUrl = url
        if FileManager.default.fileExists(atPath: modelUrl.appending(component: "vocab.json").path(percentEncoded: false)) {
            self.baseUrl = modelUrl
        } else if let baseUrl {
            self.baseUrl = baseUrl
        } else  {
            self.baseUrl = modelUrl.deletingLastPathComponent()
        }
        
        let metadata = try CoreMLMetadata.metadataForModel(at: url)
        self.maxInputLength = metadata.inputSchema[0].shape.last!
        if let hiddenSizeString = metadata.userDefinedMetadata?["hidden_size"], let hiddenSize = Int(hiddenSizeString) {
            self.hiddenSize = hiddenSize
        } else if metadata.mlProgramOperationTypeHistogram["Ios16.gelu"] == 23 {
            // Stable Diffusion 2.X TextEncoder
            self.hiddenSize = 1024
        } else {
            self.hiddenSize = 768
        }
        
        let configuration = configuration ?? metadata.attentionImplementation.preferredModelConfiguration
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }
    
    /// Unload the underlying model to free up memory
    public func unloadResources(clearCache: Bool = false) {
        model.unloadResources()
        if clearCache {
            cache.clear()
        }
    }

    /// Encode input text/string
    ///
    ///  - Parameters:
    ///     - text: Input text to be tokenized and then embedded
    ///  - Returns: Embedding representing the input text
    public func encode(_ text: String) throws -> (MLShapedArray<Float32>, MLShapedArray<Float32>) {
        if let cachedEmbeddings = cache[text] {
            return cachedEmbeddings
        }
        
        // Tokenize, padding to the expected length
        var (tokens, ids, weights) = tokenizer.tokenizeWithWeights(text, minCount: maxInputLength)

        // Truncate if necessary
        if ids.count > maxInputLength {
            tokens = tokens.dropLast(tokens.count - maxInputLength)
            ids = ids.dropLast(ids.count - maxInputLength)
            weights = weights.dropLast(weights.count - maxInputLength)
            let truncated = tokenizer.decode(tokens: tokens)
            print("Needed to truncate input '\(text)' to '\(truncated)'")
        }

        // Use the model to generate the embedding
        let embeddings = try encode(ids: ids, weights: weights)
        cache[text] = embeddings
        return embeddings
    }

    func encode(ids: [Int], weights: [Float]?) throws -> (MLShapedArray<Float32>, MLShapedArray<Float32>) {
        let result = try model.perform { model in
            let inputDescription = model.modelDescription.inputDescriptionsByName.first!.value
            let inputName = inputDescription.name
            let inputShape = inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
            
            let floatIds = ids.map { Float32($0) }
            let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
            let inputFeatures = try! MLDictionaryFeatureProvider(
                dictionary: [inputName: MLMultiArray(inputArray)])
            
            return try model.prediction(from: inputFeatures)
        }
        let pooledOutputsFeature = result.featureValue(for: "pooled_outputs")
        let pooledOutputs = MLShapedArray<Float32>(converting: pooledOutputsFeature!.multiArrayValue!)
        
        let embeddingFeature = result.featureValue(for: "last_hidden_state")
        var textEmbeddings = MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!)
        if let weights {
            let shape = textEmbeddings.shape
            let previousMean = textEmbeddings.scalars.withUnsafeBufferPointer({ buffer in
                buffer.reduce(0, +)
            }) / Float(textEmbeddings.scalars.count)
            let newEmbeddings = weights.enumerated().map { index, weight in
                MLShapedArray(
                    scalars: textEmbeddings[0][index].scalars.map { Float32($0 * weight) },
                    shape: textEmbeddings[0][index].shape
                )
            }
            textEmbeddings = MLShapedArray(concatenating: newEmbeddings, alongAxis: 0)
            let currentMean = textEmbeddings.scalars.withUnsafeBufferPointer({ buffer in
                buffer.reduce(0, +)
            }) / Float(textEmbeddings.scalars.count)
            let meanFactor = Float32(previousMean / currentMean)
            textEmbeddings = MLShapedArray(unsafeUninitializedShape: shape) { scalars, _ in
                textEmbeddings.withUnsafeShapedBufferPointer { embeddings, _, _ in
                    for i in 0..<embeddings.count {
                        scalars.initializeElement(at: i, to: embeddings[i] * meanFactor)
                    }
                }
            }
        }
        return (pooledOutputs, textEmbeddings)
    }
}
