//
//  ManagedMLModel.swift
//  
//
//  Created by Guillermo Cique Fernández on 23/5/23.
//

import CoreML

/// A class to manage and gate access to a Core ML model
///
/// It will automatically load a model into memory when needed or requested
/// It allows one to request to unload the model from memory
final class ManagedMLModel {
    /// The location of the model
    let modelURL: URL
    
    /// The configuration to be used when the model is loaded
    var configuration: MLModelConfiguration {
        didSet { unloadResources() }
    }

    /// The loaded model (when loaded)
    var loadedModel: MLModel?

    /// Queue to protect access to loaded model
    let queue: DispatchQueue

    /// Create a managed model given its location and desired loaded configuration
    ///
    /// - Parameters:
    ///     - url: The location of the model
    ///     - configuration: The configuration to be used when the model is loaded/used
    /// - Returns: A managed model that has not been loaded
    init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.modelURL = url
        self.configuration = configuration
        self.loadedModel = nil
        self.queue = DispatchQueue(label: "managed.\(url.lastPathComponent)")
    }

    /// Unload the model if it was loaded
    func unloadResources() {
        queue.sync {
            loadedModel = nil
        }
    }

    /// Perform an operation with the managed model via a supplied closure.
    ///  The model will be loaded and supplied to the closure and should only be
    ///  used within the closure to ensure all resource management is synchronized
    ///
    /// - Parameters:
    ///     - body: Closure which performs and action on a loaded model
    /// - Returns: The result of the closure
    /// - Throws: An error if the model cannot be loaded or if the closure throws
    func perform<R>(_ body: (MLModel) throws -> R) throws -> R {
        return try queue.sync {
            try autoreleasepool {
                try loadModel()
                return try body(loadedModel!)
            }
        }
    }

    private func loadModel() throws {
        if loadedModel == nil {
            loadedModel = try MLModel(contentsOf: modelURL, configuration: configuration)
        }
    }
}
