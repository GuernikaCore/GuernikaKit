//
//  CoreMLMetadata.swift
//  
//
//  Created by Guillermo Cique Fernández on 9/4/23.
//

import CoreML
import Foundation

struct CoreMLMetadata: Decodable {
    let mlProgramOperationTypeHistogram: [String: Int]
    let inputSchema: [Input]
    let userDefinedMetadata: [String: String]?
    
    var attentionImplementation: AttentionImplementation {
        mlProgramOperationTypeHistogram["Ios16.einsum"] != nil ? .splitEinsum : .original
    }
    
    var hiddenSize: Int? {
        guard let shape = inputSchema[name: "encoder_hidden_states"]?.shape else { return nil }
        guard shape.count == 4 else { return nil }
        return shape[1]
    }
    
    static func metadataForModel(at url: URL) throws -> CoreMLMetadata {
        let jsonData = try Data(contentsOf: url.appendingPathComponent("metadata.json"))
        let metadatas = try JSONDecoder().decode([CoreMLMetadata].self, from: jsonData)
        return metadatas.first!
    }
    
    struct Input: Decodable {
        let name: String
        let shape: [Int]
        let shapeRange: [[Int]]
        let hasShapeFlexibility: Bool
        
        enum CodingKeys: CodingKey {
            case name
            case shape
            case shapeRange
            case hasShapeFlexibility
        }
        
        init(from decoder: Swift.Decoder) throws {
            let container: KeyedDecodingContainer<CodingKeys> = try decoder.container(keyedBy: CodingKeys.self)
            self.name = try container.decode(String.self, forKey: .name)
            let shapeString = try container.decode(String.self, forKey: .shape)
            let shape = shapeString.shapeArray
            self.shape = shape
            
            if let shapeFlexibilityString = try container.decodeIfPresent(String.self, forKey: .hasShapeFlexibility),
               shapeFlexibilityString == "1" {
                self.hasShapeFlexibility = true
                let shapeRangeString = try container.decode(String.self, forKey: .shapeRange)
                self.shapeRange = shapeRangeString.shapeRangeArray
            } else {
                self.hasShapeFlexibility = false
                self.shapeRange = shape.map { [$0] }
            }
        }
    }
}

fileprivate extension String {
    var shapeArray: [Int] {
        self[self.index(after: startIndex)..<index(before: endIndex)]
                        .split(separator: ", ")
                        .compactMap { Int($0) }
    }
    
    var shapeRangeArray: [[Int]] {
        // Remove initial [[ and final ]]
        self[self.index(startIndex, offsetBy: 2)..<index(endIndex, offsetBy: -2)]
            .split(separator: "], [")
            .compactMap {
                $0.split(separator: ", ")
                    .compactMap { Int($0) }
            }
    }
}

extension Array where Element == CoreMLMetadata.Input {
    subscript(name name: String) -> Element? {
        return first(where: { $0.name == name })
    }
}
