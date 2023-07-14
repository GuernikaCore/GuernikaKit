//
//  BPETokenizer.swift
//
//
//  Created by Guillermo Cique Fernández on 23/5/23.
//

import Foundation

/// A tokenizer based on byte pair encoding.
public struct BPETokenizer {
    /// A dictionary that maps pairs of tokens to the rank/order of the merge.
    let merges: [TokenPair : Int]

    /// A dictionary from of tokens to identifiers.
    let vocabulary: [String: Int]

    /// The start token.
    let startToken: String = "<|startoftext|>"

    /// The end token.
    let endToken: String = "<|endoftext|>"

    /// The token used for padding, value 0
    var padToken: String = "!"

    /// The unknown token.
    let unknownToken: String = "<|endoftext|>"

    var unknownTokenID: Int {
        vocabulary[unknownToken, default: 0]
    }
    
    public internal(set) var addedVocab: [String: Int] = [:]

    /// Creates a tokenizer.
    ///
    /// - Parameters:
    ///   - merges: A dictionary that maps pairs of tokens to the rank/order of the merge.
    ///   - vocabulary: A dictionary from of tokens to identifiers.
    public init(merges: [TokenPair: Int], vocabulary: [String: Int], addedVocab: [String: Int] = [:]) {
        self.merges = merges
        self.vocabulary = vocabulary
        self.addedVocab = addedVocab
    }

    /// Creates a tokenizer by loading merges and vocabulary from URLs.
    ///
    /// - Parameters:
    ///   - mergesUrl: The URL of a text file containing merges.
    ///   - vocabularyUrl: The URL of a JSON file containing the vocabulary.
    ///   - addedVocabUrl: The URL of a JSON file containing the added vocabulary.
    public init(mergesUrl: URL, vocabularyUrl: URL, addedVocabUrl: URL?) throws {
        self.merges = try Self.readMerges(url: mergesUrl)
        self.vocabulary = try Self.readVocabulary(url: vocabularyUrl)
        if let addedVocabUrl, FileManager.default.fileExists(atPath: addedVocabUrl.path(percentEncoded: false)) {
            self.addedVocab = try Self.readVocabulary(url: addedVocabUrl)
        } else {
            self.addedVocab = [:]
        }
    }

    /// Tokenizes an input string.
    ///
    /// - Parameters:
    ///   - input: A string.
    ///   - minCount: The minimum number of tokens to return.
    /// - Returns: An array of tokens and an array of token identifiers.
    public func tokenize(_ input: String, minCount: Int? = nil) -> (tokens: [String], tokenIDs: [Int]) {
        let (tokens, ids, _) = tokenizeWithWeights(input, minCount: minCount)
        return (tokens: tokens, tokenIDs: ids)
    }

    /// Returns the token identifier for a token.
    public func tokenID(for token: String) -> Int? {
        vocabulary[token] ?? addedVocab[token]
    }

    /// Returns the token for a token identifier.
    public func token(id: Int) -> String? {
        vocabulary.first(where: { $0.value == id })?.key ?? addedVocab.first(where: { $0.value == id })?.key
    }

    /// Decodes a sequence of tokens into a fully formed string
    public func decode(tokens: [String]) -> String {
        String(tokens.joined())
            .replacingOccurrences(of: "</w>", with: " ")
            .replacingOccurrences(of: startToken, with: "")
            .replacingOccurrences(of: endToken, with: "")
    }
    
    func splitSpecialTokens(input: String) -> [String] {
        // Sort tokens by length, split before with the longest one
        let specialTokens = addedVocab.keys.sorted(by: { $0.count > $1.count })
        let result = aux_splitSpecialTokens(input: input, tokens: specialTokens)
            .filter { !$0.isEmpty }
        return result
    }
    
    private func aux_splitSpecialTokens(input: String, tokens: [String]) -> [String] {
        guard let token = tokens.first else { return [input.lowercased()] }
        let leftoverTokens = Array(tokens.dropFirst())
        // Split with current token and split results with leftover tokens
        let splits = input.split(separator: token, omittingEmptySubsequences: false)
            .map { aux_splitSpecialTokens(input: String($0), tokens: leftoverTokens) }
        // Join splits with token used to split
        return Array(splits.joined(separator: [token]))
    }

    /// Encode an input string to a sequence of tokens
    func encode(input: String) -> [String] {
        let splits = splitSpecialTokens(input: input)
        return splits.flatMap { split in
            // Don't encode special tokens
            guard addedVocab[split] == nil else { return maybeConvertToMultiVector(token: split) }
            let pattern = #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#
            let regex = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
            let range = NSRange(location: 0, length: split.utf16.count)
            let words = regex.matches(in: split, options: [], range: range)
                .map { String(split[Range($0.range, in: split)!]) }
                .map { token -> String in
                    return Array(token.utf8).compactMap { BPETokenizer.byteEncoder[$0] }.joined()
                }
            return words.flatMap({ encode(word: String($0)) })
        }
    }
    
    /**
     Maybe convert a token into a "multi vector"-compatible token. If the token corresponds
     to a multi-vector textual inversion embedding, this function will process the token so that it
     is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt
     is a single vector, the input prompt is simply returned.
     */
    func maybeConvertToMultiVector(token: String) -> [String] {
        guard let regex = try? Regex("\(token)_\\d") else { return [token] }
        return [token] + addedVocab.keys.filter { $0.wholeMatch(of: regex) != nil }
            .sorted { $0.localizedStandardCompare($1) == .orderedAscending }
    }

    /// Encode a single word into a sequence of tokens
    func encode(word: String) -> [String] {
        var tokens = word.lowercased().map { String($0) }
        if let last = tokens.indices.last {
            tokens[last] = tokens[last] + "</w>"
        }

        while true {
            let pairs = pairs(for: tokens)
            let canMerge = pairs.filter { merges[$0] != nil }

            if canMerge.isEmpty {
                break
            }

            // If multiple merges are found, use the one with the lowest rank
            let shouldMerge = canMerge.min { merges[$0]! < merges[$1]! }!
            tokens = update(tokens, merging: shouldMerge)
        }
        return tokens
    }

    /// Get  the set of adjacent pairs / bigrams from a sequence of tokens
    func pairs(for tokens: [String]) -> Set<TokenPair> {
        guard tokens.count > 1 else {
            return Set()
        }

        var pairs = Set<TokenPair>(minimumCapacity: tokens.count - 1)
        var prev = tokens.first!
        for current in tokens.dropFirst() {
            pairs.insert(TokenPair(prev, current))
            prev = current
        }
        return pairs
    }

    /// Update the sequence of tokens by greedily merging instance of a specific bigram
    func update(_ tokens: [String], merging bigram: TokenPair) -> [String] {
        guard tokens.count > 1 else {
            return []
        }

        var newTokens = [String]()
        newTokens.reserveCapacity(tokens.count - 1)

        var index = 0
        while index < tokens.count {
            let remainingTokens = tokens[index...]
            if let startMatchIndex = remainingTokens.firstIndex(of: bigram.first) {
                // Found a possible match, append everything before it
                newTokens.append(contentsOf: tokens[index..<startMatchIndex])

                if index < tokens.count - 1 && tokens[startMatchIndex + 1] == bigram.second {
                    // Full match, merge
                    newTokens.append(bigram.first + bigram.second)
                    index = startMatchIndex + 2
                } else {
                    // Only matched the first, no merge
                    newTokens.append(bigram.first)
                    index = startMatchIndex + 1
                }
            } else {
                // Didn't find any more matches, append the rest unmerged
                newTokens.append(contentsOf: remainingTokens)
                break
            }
        }
        return newTokens
    }
}

extension BPETokenizer {
    /// A hashable tuple of strings
    public struct TokenPair: Hashable {
        let first: String
        let second: String

        init(_ first: String, _ second: String) {
            self.first = first
            self.second = second
        }
    }
}
