//
//  BPETokenizer+WeightedTokens.swift
//  
//
//  Created by Guillermo Cique Fernández on 22/3/23.
//

import Foundation

extension BPETokenizer {
    public func tokenizeWithWeights(
        _ input: String,
        minCount: Int? = nil
    ) -> (tokens: [String], tokenIDs: [Int], weights: [Float]) {
        var tokens: [String] = []
        var weights: [Float] = []
        
        tokens.append(startToken)
        weights.append(1)
        let normalizedInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let textsAndWeights = parsePromptAttention(input: normalizedInput)
        for (text, weight) in textsAndWeights {
            let textTokens = encode(input: text)
            tokens.append(contentsOf: textTokens)
            weights.append(contentsOf: [Float](repeating: weight, count: textTokens.count))
        }
        tokens.append(endToken)
        weights.append(1)
        
        // Pad if there was a min length specified
        if let minLen = minCount, minLen > tokens.count {
            tokens.append(contentsOf: repeatElement(padToken, count: minLen - tokens.count))
            weights.append(contentsOf: repeatElement(1, count: minLen - weights.count))
        }
        
        let ids = tokens.map({ vocabulary[$0] ?? addedVocab[$0, default: unknownTokenID] })
        return (tokens, ids, weights)
    }
    
    func parsePromptAttention(input: String) -> [(String, Float)] {
        let pattern = #"\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|]|[^\\()\[\]:]+|:"#
        let regex = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        let range = NSRange(location: 0, length: input.utf16.count)
        let matches = regex.matches(in: input, options: [], range: range)
        
        var result: [(String, Float)] = []
        var roundBrakets: [Int] = []
        var squareBrakets: [Int] = []
        
        let roundBracketMultiplier: Float = 1.1
        let squareBracketMultiplier: Float = 1 / 1.1
        
        func multiplyRange(start: Int, multiplier: Float) {
            for p in (start..<result.count) {
                result[p].1 *= multiplier
            }
        }
        
        for match in matches {
            let text = Range(match.range(at: 0), in: input).map { String(input[$0]) } ?? ""
            let weight = Range(match.range(at: 1), in: input).map { String(input[$0]) }
            if text.starts(with: "\\") {
                result.append((String(text.dropFirst()), 1))
            } else if text == "(" {
                roundBrakets.append(result.count)
            } else if text == "[" {
                squareBrakets.append(result.count)
            } else if let weight, let start = roundBrakets.popLast() {
                multiplyRange(start: start, multiplier: Float(weight) ?? 1)
            } else if text == ")", let start = roundBrakets.popLast() {
                multiplyRange(start: start, multiplier: roundBracketMultiplier)
            } else if text == "]", let start = squareBrakets.popLast() {
                multiplyRange(start: start, multiplier: squareBracketMultiplier)
            } else {
                result.append((text, 1))
            }
        }
        
        for pos in roundBrakets {
            multiplyRange(start: pos, multiplier: roundBracketMultiplier)
        }
        
        for pos in squareBrakets {
            multiplyRange(start: pos, multiplier: squareBracketMultiplier)
        }
        
        if result.isEmpty {
            result.append(("", 1))
        }
        
        var index = 0
        while index + 1 < result.count {
            if result[index].1 == result[index + 1].1 {
                result[index].0 += result[index + 1].0
                result.remove(at: index + 1)
            } else {
                index += 1
            }
        }
        
        return result
    }
}
