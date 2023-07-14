//
//  ComputeUnits.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import CoreML
import Foundation

public enum ComputeUnits: String, CaseIterable, Identifiable, CustomStringConvertible {
    case auto
    case cpuAndNeuralEngine
    case cpuAndGPU
    case all
    
    public var id: String { rawValue }
    
    public var description: String {
        switch self {
        case .auto: return "Auto"
        case .cpuAndNeuralEngine: return "CPU and Neural Engine"
        case .cpuAndGPU: return "CPU and GPU"
        case .all: return "All"
        }
    }
    
    public var shortDescription: String {
        switch self {
        case .auto: return "Auto"
        case .cpuAndNeuralEngine: return "CPU & NE"
        case .cpuAndGPU: return "CPU & GPU"
        case .all: return "All"
        }
    }
    
    func mlComputeUnits(for attentionImplementation: AttentionImplementation) -> MLComputeUnits {
        switch self {
        case .auto: return attentionImplementation.preferredComputeUnits
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuAndGPU: return .cpuAndGPU
        case .all: return .all
        }
    }
}
