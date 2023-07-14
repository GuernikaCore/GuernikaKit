//
//  SampleTimer.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import Foundation

/// A utility for timing events and tracking time statistics
///
/// Typical usage
/// ```
/// let timer: SampleTimer
///
/// for i in 0...<iterationCount {
///     timer.start()
///     doStuff()
///     timer.stop()
/// }
///
/// print(String(format: "mean: %.2f, var: %.2f",
///              timer.mean, timer.variance))
/// ```
public final class SampleTimer: Codable {
    var startTime: CFAbsoluteTime?
    var sum: Double = 0.0
    var sumOfSquares: Double = 0.0
    var count = 0
    var samples: [Double] = []

    public init() {}

    /// Start a sample, noting the current time
    public func start() {
        startTime = CFAbsoluteTimeGetCurrent()
    }
    
    public func reset() {
        sum = 0.0
        sumOfSquares = 0.0
        count = 0
        samples = []
    }

    // Stop a sample and record the elapsed time
    @discardableResult public func stop() -> Double {
        guard let startTime = startTime else {
            return 0
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        sum += elapsed
        sumOfSquares += elapsed * elapsed
        count += 1
        samples.append(elapsed)
        return elapsed
    }

    /// Mean of all sampled times
    public var mean: Double { sum / Double(count) }

    /// Variance of all sampled times
    public var variance: Double {
        guard count > 1 else {
            return 0.0
        }
        return sumOfSquares / Double(count - 1) - mean * mean
    }

    /// Standard deviation of all sampled times
    public var stdev: Double { variance.squareRoot() }

    /// Median of all sampled times
    public var median: Double {
        let sorted = samples.sorted()
        let (q, r) = sorted.count.quotientAndRemainder(dividingBy: 2)
        if r == 0 {
            return (sorted[q] + sorted[q - 1]) / 2.0
        } else {
            return Double(sorted[q])
        }
    }

    public var allSamples: [Double] {
        samples
    }
}
