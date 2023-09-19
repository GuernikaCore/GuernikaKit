//
//  DiffusionPipeline.swift
//
//
//  Created by Guillermo Cique Fernández on 14/3/23.
//

import CoreML
import Foundation

public protocol DiffusionPipeline: AnyObject, Identifiable {
    var baseUrl: URL { get }
    
    var computeUnits: ComputeUnits { get set }
    
    var sampleSize: CGSize { get }
    var minimumSize: CGSize { get }
    var maximumSize: CGSize { get }
    
    func unloadResources()
}

extension DiffusionPipeline {
    public var id: URL { baseUrl }
    
    public var name: String { baseUrl.lastPathComponent }
    
    public var allowsVariableSize: Bool { minimumSize != maximumSize }
}

/// Sampling progress details
public struct DiffusionProgress {
    public let pipeline: any DiffusionPipeline
    public let input: SampleInput
    public let step: Int
    public let stepCount: Int
    public let currentLatentSample: MLShapedArray<Float32>
}

public enum DiffusionPipelineError: LocalizedError {
    case invalidPipeline
}
