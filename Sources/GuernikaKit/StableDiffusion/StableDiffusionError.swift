//
//  StableDiffusionError.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import Foundation

public enum StableDiffusionError: LocalizedError {
    case encoderMissing
    case inputMissing
    case incompatibleTextEncoder
    case incompatibleControlNet
    case missingInputs
    
    public var errorDescription: String? {
        switch self {
        case .inputMissing:
            return "Some required innputs were missing"
        case .encoderMissing:
            return "No encoder was found"
        case .incompatibleTextEncoder:
            return "This text encoder is not compatible with this model"
        case .incompatibleControlNet:
            return "This ControlNet is not compatible with this model"
        case .missingInputs:
            return "Necessary inputs missing"
        }
    }
}
