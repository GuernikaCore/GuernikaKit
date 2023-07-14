//
//  AttentionImplementation.swift
//  
//
//  Created by Guillermo Cique Fernández on 14/3/23.
//

import CoreML
import Foundation

public enum AttentionImplementation: String, Decodable, CustomStringConvertible {
    case original = "ORIGINAL"
    case splitEinsum = "SPLIT_EINSUM"
    
    public var description: String {
        switch self {
        case .original: return "Original"
        case .splitEinsum: return "Split einsum"
        }
    }
    
    public var preferredComputeUnits: MLComputeUnits {
        #if arch(arm64)
        switch self {
        case .original:
            return .cpuAndGPU
        case .splitEinsum:
            return .cpuAndNeuralEngine
        }
        #else
        return .cpuAndGPU
        #endif
    }
    
    public var preferredModelConfiguration: MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = preferredComputeUnits
        return configuration
    }
}
