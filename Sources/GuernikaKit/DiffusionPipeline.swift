//
//  DiffusionPipeline.swift
//
//
//  Created by Guillermo Cique Fernández on 14/3/23.
//

import Foundation

public protocol DiffusionPipeline: AnyObject, Identifiable {
    var baseUrl: URL { get }
    
    var computeUnits: ComputeUnits { get set }
    
    func unloadResources()
}

extension DiffusionPipeline {
    public var id: URL { baseUrl }
    
    public var name: String { baseUrl.lastPathComponent }
}

public enum DiffusionPipelineError: LocalizedError {
    case invalidPipeline
}
