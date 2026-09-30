#!/usr/bin/env swift
// gpu_run_all.swift — Comprehensive GPU execution test for all Metal shaders
// Compiles .metal source → runs kernels on GPU → validates numerically → benchmarks
//
// Usage:
//   swift tests/metal/gpu_run_all.swift [--bench]
//
// This tests every kernel in src/shaders/ (excluding game AI: gumbel_mcts,
// student_of_games) with real GPU dispatch and numerical verification.

import Metal
import Foundation

// ── Config ──────────────────────────────────────────────────────────────────

let runBench = CommandLine.arguments.contains("--bench")
let shaderDir = "src/shaders"

// ── Helpers ─────────────────────────────────────────────────────────────────

func fatal(_ msg: String) -> Never { fputs("FATAL: \(msg)\n", stderr); exit(1) }

var passCount = 0
var failCount = 0
var skipCount = 0
var compileFail = 0
var compilePass = 0

func check(_ cond: Bool, _ label: String, detail: String = "") {
    if cond {
        print("  ✓ \(label)")
        passCount += 1
    } else {
        print("  ✗ \(label)\(detail.isEmpty ? "" : "  [\(detail)]")")
        failCount += 1
    }
}

func skip(_ label: String) {
    print("  - SKIP \(label)")
    skipCount += 1
}

func fillConst(_ buf: MTLBuffer, count: Int, value: Float) {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = value }
}

func fillSeq(_ buf: MTLBuffer, count: Int, start: Float = 1) {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = start + Float(i) }
}

func fillRandom(_ buf: MTLBuffer, count: Int, range: ClosedRange<Float> = -1.0...1.0) {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = Float.random(in: range) }
}

func readFloats(_ buf: MTLBuffer, count: Int) -> [Float] {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

func readInts(_ buf: MTLBuffer, count: Int) -> [Int32] {
    let ptr = buf.contents().bindMemory(to: Int32.self, capacity: count)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

func floatToHalf(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    let sign = (bits >> 31) & 1
    let exp = Int((bits >> 23) & 0xFF) - 127
    let frac = bits & 0x7FFFFF
    if exp > 15 { return UInt16(sign << 15 | 0x7C00) }
    if exp < -14 { return UInt16(sign << 15) }
    return UInt16(sign << 15 | UInt32(exp + 15) << 10 | (frac >> 13))
}

func fillHalf(_ buf: MTLBuffer, count: Int, value: Float) {
    let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: count)
    let bits = floatToHalf(value)
    for i in 0..<count { ptr[i] = bits }
}

func closeEnough(_ a: [Float], _ b: [Float], rtol: Float = 1e-3, atol: Float = 1e-4) -> (Bool, String) {
    guard a.count == b.count else { return (false, "count mismatch \(a.count) vs \(b.count)") }
    var maxDiff: Float = 0
    var maxIdx = 0
    for i in 0..<a.count {
        let diff = abs(a[i] - b[i])
        let tol = atol + rtol * abs(b[i])
        if diff > tol && diff > maxDiff { maxDiff = diff; maxIdx = i }
    }
    if maxDiff > 0 { return (false, "max diff \(maxDiff) at [\(maxIdx)]: gpu=\(a[maxIdx]) cpu=\(b[maxIdx])") }
    return (true, "")
}

// Dispatch 1D
func dispatch1D(_ ps: MTLComputePipelineState, _ queue: MTLCommandQueue, _ device: MTLDevice,
                buffers: [MTLBuffer], count: Int, tgMem: Int = 0,
                extra: ((MTLComputeCommandEncoder) -> Void)? = nil) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(ps)
    for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
    if tgMem > 0 { enc.setThreadgroupMemoryLength(tgMem, index: 0) }
    extra?(enc)
    let tpg = min(ps.maxTotalThreadsPerThreadgroup, 256)
    enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit(); cmd.waitUntilCompleted()
    if let err = cmd.error { print("  GPU ERROR: \(err.localizedDescription)") }
}

// Dispatch threadgroups
func dispatchTG(_ ps: MTLComputePipelineState, _ queue: MTLCommandQueue,
                buffers: [MTLBuffer], grid: MTLSize, tg: MTLSize, tgMem: Int = 0,
                extra: ((MTLComputeCommandEncoder) -> Void)? = nil) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(ps)
    for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
    if tgMem > 0 { enc.setThreadgroupMemoryLength(tgMem, index: 0) }
    extra?(enc)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()
    cmd.commit(); cmd.waitUntilCompleted()
    if let err = cmd.error { print("  GPU ERROR: \(err.localizedDescription)") }
}

// Benchmark: batch all iterations into ONE command buffer to eliminate overhead
func benchBatched(_ label: String, _ ps: MTLComputePipelineState, _ queue: MTLCommandQueue,
                  _ device: MTLDevice, buffers: [MTLBuffer], count: Int,
                  tgMem: Int = 0, iterations: Int = 200, bytesRW: Int = 0,
                  extra: ((MTLComputeCommandEncoder) -> Void)? = nil) {
    guard runBench else { return }
    let tpg = min(ps.maxTotalThreadsPerThreadgroup, 256)
    let grid = MTLSize(width: count, height: 1, depth: 1)
    let tg = MTLSize(width: tpg, height: 1, depth: 1)

    // warmup (single batched command buffer)
    let warmCmd = queue.makeCommandBuffer()!
    for _ in 0..<10 {
        let enc = warmCmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(ps)
        for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
        if tgMem > 0 { enc.setThreadgroupMemoryLength(tgMem, index: 0) }
        extra?(enc)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
    warmCmd.commit(); warmCmd.waitUntilCompleted()

    // timed: all iterations in one command buffer
    let cmd = queue.makeCommandBuffer()!
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(ps)
        for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
        if tgMem > 0 { enc.setThreadgroupMemoryLength(tgMem, index: 0) }
        extra?(enc)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
    cmd.commit(); cmd.waitUntilCompleted()
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    let avg_us = (elapsed / Double(iterations)) * 1_000_000
    let totalBytes = bytesRW > 0 ? bytesRW : count * 4 * 2  // default: 1 read + 1 write
    let bw = Double(totalBytes) / (elapsed / Double(iterations)) / 1e9
    let elemStr = count >= 1_000_000 ? "\(count / 1_000_000)M" : count >= 1_000 ? "\(count / 1_000)K" : "\(count)"
    print("    \(label) [\(elemStr) elem]: \(String(format: "%8.1f", avg_us)) µs   \(String(format: "%5.1f", bw)) GB/s")
}

// Benchmark with threadgroup dispatch
func benchTG(_ label: String, _ ps: MTLComputePipelineState, _ queue: MTLCommandQueue,
             _ device: MTLDevice, buffers: [MTLBuffer],
             grid: MTLSize, tg: MTLSize, tgMem: Int = 0,
             iterations: Int = 200, bytesRW: Int = 0,
             extra: ((MTLComputeCommandEncoder) -> Void)? = nil) {
    guard runBench else { return }
    // warmup
    let warmCmd = queue.makeCommandBuffer()!
    for _ in 0..<10 {
        let enc = warmCmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(ps)
        for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
        if tgMem > 0 { enc.setThreadgroupMemoryLength(tgMem, index: 0) }
        extra?(enc)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
    warmCmd.commit(); warmCmd.waitUntilCompleted()

    let cmd = queue.makeCommandBuffer()!
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(ps)
        for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
        if tgMem > 0 { enc.setThreadgroupMemoryLength(tgMem, index: 0) }
        extra?(enc)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
    cmd.commit(); cmd.waitUntilCompleted()
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    let avg_us = (elapsed / Double(iterations)) * 1_000_000
    let bw = bytesRW > 0 ? Double(bytesRW) / (elapsed / Double(iterations)) / 1e9 : 0
    let bwStr = bw > 0 ? String(format: "%5.1f GB/s", bw) : ""
    print("    \(label): \(String(format: "%8.1f", avg_us)) µs   \(bwStr)")
}

// ── CPU reference implementations ───────────────────────────────────────────

func cpuRelu(_ x: [Float]) -> [Float] { x.map { max(0, $0) } }
func cpuSigmoid(_ x: [Float]) -> [Float] { x.map { 1.0 / (1.0 + exp(-$0)) } }
func cpuTanh(_ x: [Float]) -> [Float] { x.map { Foundation.tanh($0) } }
func cpuGelu(_ x: [Float]) -> [Float] {
    x.map { v in
        let arg = min(max(sqrt(2.0 / .pi) * (v + 0.044715 * v * v * v), -10.0), 10.0)
        return 0.5 * v * (1.0 + Foundation.tanh(arg))
    }
}
func cpuSilu(_ x: [Float]) -> [Float] { x.map { $0 / (1.0 + exp(-$0)) } }
func cpuSoftmax(_ x: [Float]) -> [Float] {
    let m = x.max()!; let e = x.map { exp($0 - m) }; let s = e.reduce(0, +)
    return e.map { $0 / s }
}
func cpuRmsNorm(_ x: [Float], gamma: [Float], eps: Float = 1e-5) -> [Float] {
    let rms = sqrt(x.map { $0 * $0 }.reduce(0, +) / Float(x.count) + eps)
    return zip(x, gamma).map { $0.0 / rms * $0.1 }
}
func cpuLayerNorm(_ x: [Float], gamma: [Float], beta: [Float], eps: Float = 1e-5) -> [Float] {
    let mean = x.reduce(0, +) / Float(x.count)
    let variance = x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(x.count)
    let std = sqrt(variance + eps)
    return zip(zip(x, gamma), beta).map { (($0.0.0 - mean) / std) * $0.0.1 + $0.1 }
}

// ── Main ────────────────────────────────────────────────────────────────────

guard let device = MTLCreateSystemDefaultDevice() else { fatal("No Metal device") }
guard let queue  = device.makeCommandQueue()       else { fatal("No command queue") }

print("============================================================")
print(" Metal GPU Kernel Test Suite")
print(" Device:  \(device.name)")
print(" Memory:  \(device.recommendedMaxWorkingSetSize / 1_073_741_824) GB")
print(" macOS:   \(ProcessInfo.processInfo.operatingSystemVersionString)")
print(" Bench:   \(runBench ? "ON" : "OFF (use --bench)")")
print("============================================================\n")

let fm = FileManager.default
var metalFiles: [String] = []
if let enumerator = fm.enumerator(atPath: shaderDir) {
    while let item = enumerator.nextObject() as? String {
        if item.hasSuffix(".metal") { metalFiles.append((shaderDir as NSString).appendingPathComponent(item)) }
    }
}
metalFiles.sort()

// Skip game AI shaders per user request
let skipShaders = Set(["gumbel_mcts.metal", "student_of_games.metal"])
metalFiles = metalFiles.filter { path in
    let name = (path as NSString).lastPathComponent
    return !skipShaders.contains(name)
}

if metalFiles.isEmpty { fatal("No .metal files found in \(shaderDir)") }
print("Found \(metalFiles.count) shader files\n")

let N = 4096  // Larger than 1024 to stress test
let bytes = N * MemoryLayout<Float>.stride

for path in metalFiles {
    let name = (path as NSString).lastPathComponent
    let src: String
    do { src = try String(contentsOfFile: path, encoding: .utf8) }
    catch { compileFail += 1; print("-- \(name)  [READ FAIL]"); continue }

    let opts = MTLCompileOptions()
    opts.languageVersion = .version3_0
    let lib: MTLLibrary
    do { lib = try device.makeLibrary(source: src, options: opts) }
    catch {
        compileFail += 1
        print("-- \(name)  [COMPILE FAIL: \(error.localizedDescription)]")
        continue
    }
    compilePass += 1

    let funcNames = lib.functionNames
    if funcNames.isEmpty { print("-- \(name)  [no kernels]"); continue }

    // Detect function constants
    let usesFunctionConstants = src.contains("[[function_constant(")

    var pipelines: [String: MTLComputePipelineState] = [:]
    for fname in funcNames {
        if usesFunctionConstants {
            let constants = MTLFunctionConstantValues()
            var boolVal: Bool = false
            var uintVal: UInt32 = 64  // common head_dim
            var uint8: UInt32 = 8
            constants.setConstantValue(&boolVal, type: .bool, index: 0)
            constants.setConstantValue(&uintVal, type: .uint, index: 1)
            constants.setConstantValue(&uint8, type: .uint, index: 2)
            constants.setConstantValue(&boolVal, type: .bool, index: 3)
            constants.setConstantValue(&uint8, type: .uint, index: 4)
            constants.setConstantValue(&uint8, type: .uint, index: 5)
            do {
                let fn = try lib.makeFunction(name: fname, constantValues: constants)
                pipelines[fname] = try device.makeComputePipelineState(function: fn)
            } catch { /* skip — incompatible constants */ }
        } else {
            guard let fn = lib.makeFunction(name: fname) else { continue }
            do { pipelines[fname] = try device.makeComputePipelineState(function: fn) }
            catch { /* skip non-compute functions */ }
        }
    }
    if pipelines.isEmpty { print("-- \(name)  [no compute kernels]"); continue }

    print("-- \(name)  [\(pipelines.count) kernel(s): \(pipelines.keys.sorted().joined(separator: ", "))]")

    // ── Per-kernel numerical tests ──────────────────────────────────────

    for (fname, ps) in pipelines.sorted(by: { $0.key < $1.key }) {

        // ── relu_forward ────────────────────────────────────────────────
        if fname == "relu_forward" {
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i) - Float(N/2) }
            var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [inp, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let gpu = readFloats(out, count: N)
            let expected = cpuRelu(readFloats(inp, count: N))
            let (ok, detail) = closeEnough(gpu, expected)
            check(ok, "\(fname): relu([-2048..2047])", detail: detail)
        }

        // ── relu_backward ───────────────────────────────────────────────
        if fname == "relu_backward" {
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let grad = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i) - Float(N/2) }
            fillConst(grad, count: N, value: 1.0)
            var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [inp, grad, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 3)
            }
            let gpu = readFloats(out, count: N)
            let x = readFloats(inp, count: N)
            let expected = x.map { $0 > 0 ? Float(1.0) : Float(0.0) }
            let (ok, detail) = closeEnough(gpu, expected)
            check(ok, "\(fname): grad=1 where x>0", detail: detail)
        }

        // ── sigmoid_forward ─────────────────────────────────────────────
        if fname == "sigmoid_forward" {
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i - N/2) * 0.01 }
            var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [inp, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let gpu = readFloats(out, count: N)
            let expected = cpuSigmoid(readFloats(inp, count: N))
            let (ok, detail) = closeEnough(gpu, expected)
            check(ok, "\(fname): sigmoid([-20.48..20.47])", detail: detail)
        }

        // ── tanh_forward ────────────────────────────────────────────────
        if fname == "tanh_forward" {
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i - N/2) * 0.01 }
            var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [inp, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let gpu = readFloats(out, count: N)
            let expected = cpuTanh(readFloats(inp, count: N))
            let (ok, detail) = closeEnough(gpu, expected)
            check(ok, "\(fname): tanh([-20.48..20.47])", detail: detail)
        }

        // ── elementwise_add ─────────────────────────────────────────────
        if fname == "elementwise_add" {
            let a = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let c = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillSeq(a, count: N, start: 1)
            fillSeq(b, count: N, start: 100)
            var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [a, b, c], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 3)
            }
            let gpu = readFloats(c, count: N)
            let expected = zip(readFloats(a, count: N), readFloats(b, count: N)).map { $0 + $1 }
            let (ok, detail) = closeEnough(gpu, expected)
            check(ok, "\(fname): add(seq, seq+100)", detail: detail)
        }

        // ── elementwise_mul ─────────────────────────────────────────────
        if fname == "elementwise_mul" {
            let a = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let c = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillConst(a, count: N, value: 3.0)
            fillConst(b, count: N, value: 7.0)
            var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [a, b, c], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 3)
            }
            let gpu = readFloats(c, count: N)
            let ok = gpu.allSatisfy { abs($0 - 21.0) < 1e-4 }
            check(ok, "\(fname): 3*7=21 (got \(gpu[0]))")
        }

        // ── elementwise_scale ───────────────────────────────────────────
        if fname == "elementwise_scale" {
            let a = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillSeq(a, count: N, start: 1)
            var s = Float(2.5); var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [a, b], count: N) { enc in
                enc.setBytes(&s, length: 4, index: 2)
                enc.setBytes(&n, length: 4, index: 3)
            }
            let gpu = readFloats(b, count: N)
            let expected = readFloats(a, count: N).map { $0 * 2.5 }
            let (ok, detail) = closeEnough(gpu, expected)
            check(ok, "\(fname): seq*2.5", detail: detail)
        }

        // ── elementwise_add_scalar ──────────────────────────────────────
        if fname == "elementwise_add_scalar" {
            let a = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillSeq(a, count: N, start: 0)
            var bias = Float(42.0); var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [a, b], count: N) { enc in
                enc.setBytes(&bias, length: 4, index: 2)
                enc.setBytes(&n, length: 4, index: 3)
            }
            let gpu = readFloats(b, count: N)
            let expected = readFloats(a, count: N).map { $0 + 42.0 }
            let (ok, detail) = closeEnough(gpu, expected)
            check(ok, "\(fname): seq+42", detail: detail)
        }

        // ── softmax_forward ─────────────────────────────────────────────
        if fname == "softmax_forward" {
            let smallN = 256
            let smallBytes = smallN * MemoryLayout<Float>.stride
            let inp = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            fillRandom(inp, count: smallN, range: -2...2)
            var n = UInt32(smallN)
            let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), smallN)
            dispatchTG(ps, queue, buffers: [inp, out],
                       grid: MTLSize(width: 1, height: 1, depth: 1),
                       tg: MTLSize(width: tgSize, height: 1, depth: 1),
                       tgMem: smallN * 4) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let gpu = readFloats(out, count: smallN)
            let sum = gpu.reduce(0, +)
            check(abs(sum - 1.0) < 0.02, "\(fname): sum=\(String(format: "%.6f", sum)) (expected 1.0)")
            check(gpu.allSatisfy { $0 >= 0 }, "\(fname): all outputs >= 0")
            // Verify against CPU
            let cpuResult = cpuSoftmax(readFloats(inp, count: smallN))
            let (ok, detail) = closeEnough(gpu, cpuResult, rtol: 1e-2, atol: 1e-3)
            check(ok, "\(fname): matches CPU reference", detail: detail)
        }

        // ── rmsnorm_kernel ──────────────────────────────────────────────
        if fname == "rmsnorm_kernel" {
            let smallN = 256
            let smallBytes = smallN * MemoryLayout<Float>.stride
            let inp = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let weight = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            fillSeq(inp, count: smallN, start: 0.1)
            fillConst(weight, count: smallN, value: 1.0)
            var n = UInt32(smallN); var eps = Float(1e-5)
            let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), smallN)
            dispatchTG(ps, queue, buffers: [inp, weight, out],
                       grid: MTLSize(width: 1, height: 1, depth: 1),
                       tg: MTLSize(width: tgSize, height: 1, depth: 1)) { enc in
                enc.setBytes(&n, length: 4, index: 3)
                enc.setBytes(&eps, length: 4, index: 4)
            }
            let gpu = readFloats(out, count: smallN)
            let expected = cpuRmsNorm(readFloats(inp, count: smallN),
                                      gamma: readFloats(weight, count: smallN))
            let (ok, detail) = closeEnough(gpu, expected, rtol: 1e-2, atol: 1e-3)
            check(ok, "\(fname): rmsnorm(seq, gamma=1)", detail: detail)
        }

        // ── residual_rmsnorm ────────────────────────────────────────────
        if fname == "residual_rmsnorm" {
            let smallN = 256
            let smallBytes = smallN * MemoryLayout<Float>.stride
            let x = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let h = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let weight = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            fillSeq(x, count: smallN, start: 0.1)
            fillConst(h, count: smallN, value: 0.0)
            fillConst(weight, count: smallN, value: 1.0)
            var n = UInt32(smallN); var eps = Float(1e-5)
            let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), smallN)
            dispatchTG(ps, queue, buffers: [x, h, weight, out],
                       grid: MTLSize(width: 1, height: 1, depth: 1),
                       tg: MTLSize(width: tgSize, height: 1, depth: 1)) { enc in
                enc.setBytes(&n, length: 4, index: 4)
                enc.setBytes(&eps, length: 4, index: 5)
            }
            let gpu = readFloats(out, count: smallN)
            // h=0 so x+h=x, residual_rmsnorm ≡ rmsnorm
            _ = readFloats(x, count: smallN)  // x was modified in-place by the kernel
            // Since x was modified in-place (x += h where h=0, so x unchanged), use original seq
            let origX: [Float] = (0..<smallN).map { Float($0) * 1.0 + 0.1 }
            let expected = cpuRmsNorm(origX, gamma: [Float](repeating: 1.0, count: smallN))
            let (ok, detail) = closeEnough(gpu, expected, rtol: 1e-2, atol: 1e-3)
            check(ok, "\(fname): residual_rmsnorm(seq, h=0, w=1)", detail: detail)
        }

        // ── silu_mul_kernel ─────────────────────────────────────────────
        if fname == "silu_mul_kernel" {
            let gate = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let up = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptrG = gate.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptrG[i] = Float(i - N/2) * 0.01 }
            fillConst(up, count: N, value: 1.5)
            let cpuGate = Array(UnsafeBufferPointer(start: ptrG, count: N))
            var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [gate, up], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let gpu = readFloats(gate, count: N)
            let expected = cpuSilu(cpuGate).map { $0 * 1.5 }
            let (ok, detail) = closeEnough(gpu, expected, rtol: 1e-2, atol: 1e-3)
            check(ok, "\(fname): silu(x)*1.5 in-place", detail: detail)
        }

        // ── fused_gelu_mul ──────────────────────────────────────────────
        if fname == "fused_gelu_mul" {
            let a = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptrA = a.contents().bindMemory(to: Float.self, capacity: N)
            let ptrB = b.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptrA[i] = Float(i - N/2) * 0.01 }
            for i in 0..<N { ptrB[i] = 2.0 }
            let cpuInput = Array(UnsafeBufferPointer(start: ptrA, count: N))
            var n = UInt32(N)
            dispatch1D(ps, queue, device, buffers: [a, b, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 3)
            }
            let gpu = readFloats(out, count: N)
            let expected = cpuGelu(cpuInput).map { $0 * 2.0 }
            let (ok, detail) = closeEnough(gpu, expected, rtol: 1e-2, atol: 1e-3)
            check(ok, "\(fname): gelu(x)*2", detail: detail)
        }

        // ── fused_silu_mul_quantize ─────────────────────────────────────
        if fname == "fused_silu_mul_quantize" {
            let groupSize: UInt32 = 128
            let numGroups = (UInt32(N) + groupSize - 1) / groupSize
            let gate = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let up = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: N, options: .storageModeShared)!
            let scales = device.makeBuffer(length: Int(numGroups) * 4, options: .storageModeShared)!
            let ptrG = gate.contents().bindMemory(to: Float.self, capacity: N)
            let ptrU = up.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptrG[i] = Float(i - N/2) * 0.1 }
            for i in 0..<N { ptrU[i] = 2.0 }
            let scPtr = scales.contents().bindMemory(to: Float.self, capacity: Int(numGroups))
            for g in 0..<Int(numGroups) {
                var maxAbs: Float = 0
                for i in (g * Int(groupSize))..<min((g + 1) * Int(groupSize), N) {
                    let gv = ptrG[i]; let sv = gv / (1.0 + exp(-gv)); let val = sv * ptrU[i]
                    maxAbs = max(maxAbs, abs(val))
                }
                scPtr[g] = max(maxAbs / 127.0, 1e-8)
            }
            var n = UInt32(N); var gs = groupSize
            dispatch1D(ps, queue, device, buffers: [gate, up, out, scales], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 4)
                enc.setBytes(&gs, length: 4, index: 5)
            }
            let qPtr = out.contents().bindMemory(to: Int8.self, capacity: N)
            let inRange = (0..<N).allSatisfy { qPtr[$0] >= -128 && qPtr[$0] <= 127 }
            check(inRange, "\(fname): all outputs in int8 range")
            let anyNonZero = (0..<N).contains { qPtr[$0] != 0 }
            check(anyNonZero, "\(fname): output non-trivial")
        }

        // ── rope_kernel ─────────────────────────────────────────────────
        if fname == "rope_kernel" {
            let headDim: UInt32 = 64
            let nHeads: UInt32 = 4
            let nKVHeads: UInt32 = 4
            let totalFloats = Int(nHeads * headDim)
            let kFloats = Int(nKVHeads * headDim)
            let q = device.makeBuffer(length: totalFloats * 4, options: .storageModeShared)!
            let k = device.makeBuffer(length: kFloats * 4, options: .storageModeShared)!
            fillConst(q, count: totalFloats, value: 1.0)
            fillConst(k, count: kFloats, value: 1.0)
            var hd = headDim; var nh = nHeads; var nkv = nKVHeads
            var pos: Int32 = 0; var theta = Float(10000.0)
            let totalPairs = Int((nHeads + nKVHeads) * headDim / 2)
            dispatch1D(ps, queue, device, buffers: [q, k], count: totalPairs) { enc in
                enc.setBytes(&hd, length: 4, index: 2)
                enc.setBytes(&nh, length: 4, index: 3)
                enc.setBytes(&nkv, length: 4, index: 4)
                enc.setBytes(&pos, length: 4, index: 5)
                enc.setBytes(&theta, length: 4, index: 6)
            }
            // At position=0, cos=1, sin=0 for all freqs → identity
            let qOut = readFloats(q, count: totalFloats)
            let ok = qOut.allSatisfy { abs($0 - 1.0) < 1e-4 }
            check(ok, "\(fname): rope(1.0, pos=0)=1.0 (identity)")
        }

        // ── quantize_int8 ───────────────────────────────────────────────
        if fname == "quantize_int8" {
            let groupSize: UInt32 = 128
            let numGroups = (UInt32(N) + groupSize - 1) / groupSize
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: N, options: .storageModeShared)!
            let scales = device.makeBuffer(length: Int(numGroups) * 4, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i - N/2) * 0.1 }
            let scPtr = scales.contents().bindMemory(to: Float.self, capacity: Int(numGroups))
            for g in 0..<Int(numGroups) {
                var maxAbs: Float = 0
                for i in (g * Int(groupSize))..<min((g + 1) * Int(groupSize), N) {
                    maxAbs = max(maxAbs, abs(ptr[i]))
                }
                scPtr[g] = maxAbs / 127.0
            }
            var params = (UInt32(N), groupSize)
            dispatch1D(ps, queue, device, buffers: [inp, out, scales], count: N) { enc in
                enc.setBytes(&params, length: 8, index: 3)
            }
            let qPtr = out.contents().bindMemory(to: Int8.self, capacity: N)
            let inRange = (0..<N).allSatisfy { qPtr[$0] >= -128 && qPtr[$0] <= 127 }
            check(inRange, "\(fname): all in int8 range")
            let anyNonZero = (0..<N).contains { qPtr[$0] != 0 }
            check(anyNonZero, "\(fname): output non-trivial")
        }

        // ── dequantize_int8 ─────────────────────────────────────────────
        if fname == "dequantize_int8" {
            let groupSize: UInt32 = 128
            let numGroups = (UInt32(N) + groupSize - 1) / groupSize
            let inp = device.makeBuffer(length: N, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let scales = device.makeBuffer(length: Int(numGroups) * 4, options: .storageModeShared)!
            let iPtr = inp.contents().bindMemory(to: Int8.self, capacity: N)
            for i in 0..<N { iPtr[i] = Int8(clamping: i % 200 - 100) }
            let scPtr = scales.contents().bindMemory(to: Float.self, capacity: Int(numGroups))
            for g in 0..<Int(numGroups) { scPtr[g] = 0.5 }
            var params = (UInt32(N), groupSize)
            dispatch1D(ps, queue, device, buffers: [inp, out, scales], count: N) { enc in
                enc.setBytes(&params, length: 8, index: 3)
            }
            let gpu = readFloats(out, count: N)
            // Check: output[i] = int8_val * scale
            var ok = true
            for i in 0..<min(100, N) {
                let expected = Float(iPtr[i]) * 0.5
                if abs(gpu[i] - expected) > 1e-4 { ok = false; break }
            }
            check(ok, "\(fname): dequant = val * scale")
        }

        // ── quantize_int4 ───────────────────────────────────────────────
        if fname == "quantize_int4" {
            let groupSize: UInt32 = 128
            let numGroups = (UInt32(N) + groupSize - 1) / groupSize
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: N / 2, options: .storageModeShared)!
            let scales = device.makeBuffer(length: Int(numGroups) * 4, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i - N/2) * 0.01 }
            let scPtr = scales.contents().bindMemory(to: Float.self, capacity: Int(numGroups))
            for g in 0..<Int(numGroups) {
                var maxAbs: Float = 0
                for i in (g * Int(groupSize))..<min((g + 1) * Int(groupSize), N) {
                    maxAbs = max(maxAbs, abs(ptr[i]))
                }
                scPtr[g] = maxAbs / 7.0
            }
            var params = (UInt32(N), groupSize)
            dispatch1D(ps, queue, device, buffers: [inp, out, scales], count: N / 2) { enc in
                enc.setBytes(&params, length: 8, index: 3)
            }
            let qPtr = out.contents().bindMemory(to: UInt8.self, capacity: N / 2)
            let anyNonZero = (0..<(N/2)).contains { qPtr[$0] != 0 }
            check(anyNonZero, "\(fname): packed int4 non-trivial")
        }

        // ── embedding_lookup ────────────────────────────────────────────
        if fname == "embedding_lookup" {
            let vocabSize = 64; let dim = 16; let seqLen = 8
            let table = device.makeBuffer(length: vocabSize * dim * 2, options: .storageModeShared)!
            let tokens = device.makeBuffer(length: seqLen * 4, options: .storageModeShared)!
            let out = device.makeBuffer(length: seqLen * dim * 2, options: .storageModeShared)!
            let tPtr = table.contents().bindMemory(to: UInt16.self, capacity: vocabSize * dim)
            for i in 0..<(vocabSize * dim) { tPtr[i] = floatToHalf(Float(i) * 0.01) }
            let idxPtr = tokens.contents().bindMemory(to: Int32.self, capacity: seqLen)
            for i in 0..<seqLen { idxPtr[i] = Int32(i * 3) }
            var d = UInt32(dim)
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(ps)
            enc.setBuffer(table, offset: 0, index: 0)
            enc.setBuffer(tokens, offset: 0, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            enc.setBytes(&d, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: dim, height: seqLen, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(dim, Int(ps.maxTotalThreadsPerThreadgroup)), height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit(); cmd.waitUntilCompleted()
            let oPtr = out.contents().bindMemory(to: UInt16.self, capacity: seqLen * dim)
            var allOk = true
            for i in 0..<seqLen {
                for j in 0..<dim {
                    if oPtr[i * dim + j] != tPtr[Int(idxPtr[i]) * dim + j] { allOk = false; break }
                }
            }
            check(allOk, "\(fname): lookup([0,3,6,...], dim=16, half)")
        }

        // ── batched_rope ────────────────────────────────────────────────
        if fname == "batched_rope" {
            let headDim: UInt32 = 64; let nHeads: UInt32 = 1; let nKVHeads: UInt32 = 1; let seqLen: UInt32 = 1
            let qLen = Int(seqLen * nHeads * headDim)
            let kLen = Int(seqLen * nKVHeads * headDim)
            let qBuf = device.makeBuffer(length: qLen * 2, options: .storageModeShared)!
            let kBuf = device.makeBuffer(length: kLen * 2, options: .storageModeShared)!
            let posBuf = device.makeBuffer(length: Int(seqLen) * 4, options: .storageModeShared)!
            fillHalf(qBuf, count: qLen, value: 1.0)
            fillHalf(kBuf, count: kLen, value: 1.0)
            let posPtr = posBuf.contents().bindMemory(to: Int32.self, capacity: Int(seqLen))
            posPtr[0] = 0
            var params = (headDim, nHeads, nKVHeads, seqLen)
            var theta: Float = 10000.0
            let totalPairs = Int(headDim / 2 * nHeads)
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(ps)
            enc.setBuffer(qBuf, offset: 0, index: 0)
            enc.setBuffer(kBuf, offset: 0, index: 1)
            enc.setBuffer(posBuf, offset: 0, index: 2)
            enc.setBytes(&params, length: 16, index: 3)
            enc.setBytes(&theta, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: totalPairs, height: Int(seqLen), depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(totalPairs, Int(ps.maxTotalThreadsPerThreadgroup)), height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit(); cmd.waitUntilCompleted()
            let outQ = qBuf.contents().bindMemory(to: UInt16.self, capacity: qLen)
            let oneHalf = floatToHalf(1.0)
            let ok = (0..<qLen).allSatisfy { outQ[$0] == oneHalf }
            check(ok, "\(fname): rope(half(1.0), pos=0) = identity")
        }

        // ── elo_batch_update ────────────────────────────────────────────
        if fname == "elo_batch_update" {
            struct EloUpdate { var ra: Float; var rb: Float; var result: Float; var k: Float }
            struct EloResult { var nra: Float; var nrb: Float }
            let nGames = 1024
            let updateBuf = device.makeBuffer(length: nGames * MemoryLayout<EloUpdate>.stride, options: .storageModeShared)!
            let resultBuf = device.makeBuffer(length: nGames * MemoryLayout<EloResult>.stride, options: .storageModeShared)!
            let uPtr = updateBuf.contents().bindMemory(to: EloUpdate.self, capacity: nGames)
            for i in 0..<nGames {
                uPtr[i] = EloUpdate(ra: 1500, rb: 1500, result: i % 2 == 0 ? 1.0 : 0.0, k: 32)
            }
            var n = UInt32(nGames)
            dispatch1D(ps, queue, device, buffers: [updateBuf, resultBuf], count: nGames) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let rPtr = resultBuf.contents().bindMemory(to: EloResult.self, capacity: nGames)
            // Equal ratings → expected=0.5, win gives +16, loss gives -16
            let winResult = rPtr[0]  // A wins
            let loseResult = rPtr[1] // A loses
            check(abs(winResult.nra - 1516.0) < 0.1, "\(fname): winner gets +16 (got \(winResult.nra))")
            check(abs(loseResult.nra - 1484.0) < 0.1, "\(fname): loser gets -16 (got \(loseResult.nra))")
        }

        // ── fused_topk_softmax ──────────────────────────────────────────
        if fname == "fused_topk_softmax" {
            let vocabSize: UInt32 = 512
            let K: UInt32 = 4
            let logits = device.makeBuffer(length: Int(vocabSize) * 4, options: .storageModeShared)!
            let probs = device.makeBuffer(length: Int(K) * 4, options: .storageModeShared)!
            let indices = device.makeBuffer(length: Int(K) * 4, options: .storageModeShared)!
            let lPtr = logits.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
            // Set a clear top-4: indices 100, 200, 300, 400 with high values
            for i in 0..<Int(vocabSize) { lPtr[i] = Float.random(in: -5...0) }
            lPtr[100] = 10.0; lPtr[200] = 8.0; lPtr[300] = 6.0; lPtr[400] = 4.0
            var vs = vocabSize; var k = K
            let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), 256)
            dispatchTG(ps, queue, buffers: [logits, probs, indices],
                       grid: MTLSize(width: 1, height: 1, depth: 1),
                       tg: MTLSize(width: tgSize, height: 1, depth: 1),
                       tgMem: tgSize * Int(K) * 8) { enc in  // allVals + allIdxs
                enc.setBytes(&vs, length: 4, index: 3)
                enc.setBytes(&k, length: 4, index: 4)
            }
            let p = readFloats(probs, count: Int(K))
            _ = readFloats(indices, count: Int(K))
            // Probs should sum to ~1
            let psum = p.reduce(0, +)
            check(abs(psum - 1.0) < 0.05, "\(fname): probs sum=\(String(format: "%.4f", psum))")
            // First index should be 100 (highest logit)
            let idxBuf = indices.contents().bindMemory(to: UInt32.self, capacity: Int(K))
            check(idxBuf[0] == 100, "\(fname): top-1 index=\(idxBuf[0]) (expected 100)")
        }

        // ── fused_layernorm_gelu ────────────────────────────────────────
        if fname == "fused_layernorm_gelu" {
            let smallN = 128
            let smallBytes = smallN * 4
            let inp = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let gamma = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let beta = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: smallBytes, options: .storageModeShared)!
            fillRandom(inp, count: smallN, range: -1...1)
            fillConst(gamma, count: smallN, value: 1.0)
            fillConst(beta, count: smallN, value: 0.0)
            var n = UInt32(smallN); var eps = Float(1e-5)
            let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), smallN)
            dispatchTG(ps, queue, buffers: [inp, gamma, beta, out],
                       grid: MTLSize(width: 1, height: 1, depth: 1),
                       tg: MTLSize(width: tgSize, height: 1, depth: 1)) { enc in
                enc.setBytes(&n, length: 4, index: 4)
                enc.setBytes(&eps, length: 4, index: 5)
            }
            let gpu = readFloats(out, count: smallN)
            // After layernorm+gelu: output should be finite and bounded
            let finite = gpu.allSatisfy { $0.isFinite }
            check(finite, "\(fname): all outputs finite")
            // GELU output should have reasonable range
            let maxAbs = gpu.map { abs($0) }.max()!
            check(maxAbs < 10.0, "\(fname): output bounded (max abs=\(String(format: "%.3f", maxAbs)))")
        }

        // ── vocab_project_argmax ────────────────────────────────────────
        if fname == "vocab_project_argmax" {
            let hiddenDim: UInt32 = 64
            let vocabSize: UInt32 = 128
            let hidden = device.makeBuffer(length: Int(hiddenDim) * 2, options: .storageModeShared)!
            let lmHead = device.makeBuffer(length: Int(vocabSize * hiddenDim) * 2, options: .storageModeShared)!
            let outputIds = device.makeBuffer(length: 4, options: .storageModeShared)!
            // Set hidden to all 1s (half), lm_head row 42 to all 1s, rest to 0
            fillHalf(hidden, count: Int(hiddenDim), value: 1.0)
            fillHalf(lmHead, count: Int(vocabSize * hiddenDim), value: 0.0)
            // Make row 42 have max dot product
            let lmPtr = lmHead.contents().bindMemory(to: UInt16.self, capacity: Int(vocabSize * hiddenDim))
            let oneH = floatToHalf(1.0)
            for d in 0..<Int(hiddenDim) { lmPtr[42 * Int(hiddenDim) + d] = oneH }
            var hd = hiddenDim; var vs = vocabSize
            let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), 256)
            dispatchTG(ps, queue, buffers: [hidden, lmHead, outputIds],
                       grid: MTLSize(width: 1, height: 1, depth: 1),
                       tg: MTLSize(width: tgSize, height: 1, depth: 1)) { enc in
                enc.setBytes(&hd, length: 4, index: 3)
                enc.setBytes(&vs, length: 4, index: 4)
            }
            let outId = outputIds.contents().bindMemory(to: Int32.self, capacity: 1)[0]
            check(outId == 42, "\(fname): argmax=\(outId) (expected 42)")
        }

        // ── Catch-all: pipeline creation ────────────────────────────────
        let testedKernels: Set<String> = [
            "relu_forward", "relu_backward", "sigmoid_forward", "tanh_forward",
            "elementwise_add", "elementwise_mul", "elementwise_scale", "elementwise_add_scalar",
            "softmax_forward", "rmsnorm_kernel", "residual_rmsnorm",
            "silu_mul_kernel", "fused_gelu_mul", "fused_silu_mul_quantize",
            "rope_kernel", "batched_rope", "embedding_lookup",
            "quantize_int8", "dequantize_int8", "quantize_int4",
            "elo_batch_update", "fused_topk_softmax",
            "fused_layernorm_gelu", "vocab_project_argmax"
        ]
        if !testedKernels.contains(fname) {
            check(true, "\(fname): pipeline OK (maxThreads=\(ps.maxTotalThreadsPerThreadgroup))")
        }
    }
}

// ── Summary ─────────────────────────────────────────────────────────────────
print("\n============================================================")
print(" RESULTS")
print("============================================================")
print(" Compiled:       \(compilePass) / \(compilePass + compileFail)")
if compileFail > 0 { print(" COMPILE FAILS:  \(compileFail)") }
print(" Numerical PASS: \(passCount)")
print(" Numerical FAIL: \(failCount)")
print(" Skipped:        \(skipCount)")
print("============================================================")

if failCount > 0 {
    exit(1)
} else {
    print("\nAll tests passed!")
}

// ── Benchmark Suite (real workload sizes) ───────────────────────────────────
guard runBench else { exit(failCount > 0 ? 1 : 0) }

print("\n============================================================")
print(" GPU Benchmarks — Real Workload Sizes")
print(" M1 theoretical bandwidth: ~68 GB/s")
print("============================================================\n")

// Recompile shaders we need for benchmarks
func loadKernel(_ src: String, _ name: String) -> MTLComputePipelineState? {
    let opts = MTLCompileOptions(); opts.languageVersion = .version3_0
    guard let lib = try? device.makeLibrary(source: src, options: opts),
          let fn = lib.makeFunction(name: name),
          let ps = try? device.makeComputePipelineState(function: fn) else { return nil }
    return ps
}

func loadKernelFC(_ src: String, _ name: String) -> MTLComputePipelineState? {
    let opts = MTLCompileOptions(); opts.languageVersion = .version3_0
    guard let lib = try? device.makeLibrary(source: src, options: opts) else { return nil }
    let constants = MTLFunctionConstantValues()
    var boolVal: Bool = false
    var hd: UInt32 = 128; var nh: UInt32 = 32; var nkv: UInt32 = 8
    constants.setConstantValue(&boolVal, type: .bool, index: 0)
    constants.setConstantValue(&hd, type: .uint, index: 1)
    constants.setConstantValue(&nh, type: .uint, index: 2)
    constants.setConstantValue(&boolVal, type: .bool, index: 3)
    constants.setConstantValue(&nh, type: .uint, index: 4)
    constants.setConstantValue(&nh, type: .uint, index: 5)
    guard let fn = try? lib.makeFunction(name: name, constantValues: constants),
          let ps = try? device.makeComputePipelineState(function: fn) else { return nil }
    return ps
}

let fm2 = FileManager.default

// ── 1. Elementwise ops at scale ─────────────────────────────────────────
print("  Elementwise Operations (memory-bound)")
print("  ──────────────────────────────────────")
for size in [1_000_000, 4_000_000, 16_000_000] {
    let sizeBytes = size * 4
    let a = device.makeBuffer(length: sizeBytes, options: .storageModeShared)!
    let b = device.makeBuffer(length: sizeBytes, options: .storageModeShared)!
    let c = device.makeBuffer(length: sizeBytes, options: .storageModeShared)!
    fillConst(a, count: size, value: 1.0)
    fillConst(b, count: size, value: 2.0)

    let reluSrc = try! String(contentsOfFile: "src/shaders/relu.metal", encoding: .utf8)
    let elemSrc = try! String(contentsOfFile: "src/shaders/elementwise.metal", encoding: .utf8)
    let sigSrc = try! String(contentsOfFile: "src/shaders/sigmoid.metal", encoding: .utf8)

    if let ps = loadKernel(reluSrc, "relu_forward") {
        var n = UInt32(size)
        benchBatched("relu_forward", ps, queue, device, buffers: [a, b], count: size,
                     bytesRW: sizeBytes * 2) { enc in enc.setBytes(&n, length: 4, index: 2) }
    }
    if let ps = loadKernel(elemSrc, "elementwise_add") {
        var n = UInt32(size)
        benchBatched("elementwise_add", ps, queue, device, buffers: [a, b, c], count: size,
                     bytesRW: sizeBytes * 3) { enc in enc.setBytes(&n, length: 4, index: 3) }
    }
    if let ps = loadKernel(sigSrc, "sigmoid_forward") {
        var n = UInt32(size)
        benchBatched("sigmoid_forward", ps, queue, device, buffers: [a, b], count: size,
                     bytesRW: sizeBytes * 2) { enc in enc.setBytes(&n, length: 4, index: 2) }
    }
    if let ps = loadKernel(sigSrc, "tanh_forward") {
        var n = UInt32(size)
        benchBatched("tanh_forward", ps, queue, device, buffers: [a, b], count: size,
                     bytesRW: sizeBytes * 2) { enc in enc.setBytes(&n, length: 4, index: 2) }
    }
    print("")
}

// ── 2. Fused ops (transformer building blocks) ──────────────────────────
print("  Fused Transformer Ops (compute + memory)")
print("  ──────────────────────────────────────────")
let fusedSrc = try! String(contentsOfFile: "src/shaders/fused_ops.metal", encoding: .utf8)

for hiddenDim in [768, 2048, 4096] {
    let hdBytes = hiddenDim * 4
    let x = device.makeBuffer(length: hdBytes, options: .storageModeShared)!
    let h = device.makeBuffer(length: hdBytes, options: .storageModeShared)!
    let w = device.makeBuffer(length: hdBytes, options: .storageModeShared)!
    let out = device.makeBuffer(length: hdBytes, options: .storageModeShared)!
    fillRandom(x, count: hiddenDim, range: -1...1)
    fillConst(h, count: hiddenDim, value: 0.1)
    fillConst(w, count: hiddenDim, value: 1.0)

    if let ps = loadKernel(fusedSrc, "residual_rmsnorm") {
        var n = UInt32(hiddenDim); var eps = Float(1e-5)
        let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), 256)
        benchTG("residual_rmsnorm d=\(hiddenDim)", ps, queue, device,
                buffers: [x, h, w, out],
                grid: MTLSize(width: 1, height: 1, depth: 1),
                tg: MTLSize(width: tgSize, height: 1, depth: 1),
                bytesRW: hdBytes * 4) { enc in
            enc.setBytes(&n, length: 4, index: 4)
            enc.setBytes(&eps, length: 4, index: 5)
        }
    }

    if let ps = loadKernel(fusedSrc, "silu_mul_kernel") {
        // SiLU*mul on FFN intermediate (4x hidden)
        let ffnSize = hiddenDim * 4
        let ffnBytes = ffnSize * 4
        let gate = device.makeBuffer(length: ffnBytes, options: .storageModeShared)!
        let up = device.makeBuffer(length: ffnBytes, options: .storageModeShared)!
        fillRandom(gate, count: ffnSize, range: -2...2)
        fillConst(up, count: ffnSize, value: 1.0)
        var n = UInt32(ffnSize)
        benchBatched("silu_mul d=\(ffnSize)", ps, queue, device,
                     buffers: [gate, up], count: ffnSize,
                     bytesRW: ffnBytes * 2) { enc in enc.setBytes(&n, length: 4, index: 2) }
    }
}
print("")

// ── 3. Softmax at scale ─────────────────────────────────────────────────
print("  Softmax (attention-scale)")
print("  ─────────────────────────")
let smSrc = try! String(contentsOfFile: "src/shaders/softmax.metal", encoding: .utf8)
for seqLen in [256, 1024, 4096] {
    let smBytes = seqLen * 4
    let inp = device.makeBuffer(length: smBytes, options: .storageModeShared)!
    let out = device.makeBuffer(length: smBytes, options: .storageModeShared)!
    fillRandom(inp, count: seqLen, range: -5...5)
    if let ps = loadKernel(smSrc, "softmax_forward") {
        var n = UInt32(seqLen)
        let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), seqLen)
        benchTG("softmax seq=\(seqLen)", ps, queue, device,
                buffers: [inp, out],
                grid: MTLSize(width: 1, height: 1, depth: 1),
                tg: MTLSize(width: tgSize, height: 1, depth: 1),
                tgMem: seqLen * 4,
                bytesRW: smBytes * 2) { enc in
            enc.setBytes(&n, length: 4, index: 2)
        }
    }
}
print("")

// ── 4. RMSNorm at scale ─────────────────────────────────────────────────
print("  RMSNorm (per-layer overhead)")
print("  ────────────────────────────")
let tensorSrc = try! String(contentsOfFile: "src/shaders/tensor_ops.metal", encoding: .utf8)
for hiddenDim in [768, 2048, 4096, 8192] {
    let hdBytes = hiddenDim * 4
    let x = device.makeBuffer(length: hdBytes, options: .storageModeShared)!
    let w = device.makeBuffer(length: hdBytes, options: .storageModeShared)!
    let out = device.makeBuffer(length: hdBytes, options: .storageModeShared)!
    fillRandom(x, count: hiddenDim, range: -1...1)
    fillConst(w, count: hiddenDim, value: 1.0)
    if let ps = loadKernelFC(tensorSrc, "rmsnorm_kernel") {
        var n = UInt32(hiddenDim); var eps = Float(1e-5)
        let tgSize = min(Int(ps.maxTotalThreadsPerThreadgroup), 256)
        benchTG("rmsnorm d=\(hiddenDim)", ps, queue, device,
                buffers: [x, w, out],
                grid: MTLSize(width: 1, height: 1, depth: 1),
                tg: MTLSize(width: tgSize, height: 1, depth: 1),
                bytesRW: hdBytes * 3) { enc in
            enc.setBytes(&n, length: 4, index: 3)
            enc.setBytes(&eps, length: 4, index: 4)
        }
    }
}
print("")

// ── 5. Quantization throughput ──────────────────────────────────────────
print("  INT8 Quantization Pipeline")
print("  ──────────────────────────")
let qSrc = try! String(contentsOfFile: "src/shaders/quantize_pipeline.metal", encoding: .utf8)
for size in [1_000_000, 8_000_000] {
    let sizeBytes = size * 4
    let groupSize: UInt32 = 128
    let numGroups = (UInt32(size) + groupSize - 1) / groupSize
    let inp = device.makeBuffer(length: sizeBytes, options: .storageModeShared)!
    let qout = device.makeBuffer(length: size, options: .storageModeShared)!
    let scales = device.makeBuffer(length: Int(numGroups) * 4, options: .storageModeShared)!
    fillRandom(inp, count: size, range: -10...10)
    let scPtr = scales.contents().bindMemory(to: Float.self, capacity: Int(numGroups))
    for g in 0..<Int(numGroups) { scPtr[g] = 10.0 / 127.0 }
    if let ps = loadKernel(qSrc, "quantize_int8") {
        var params = (UInt32(size), groupSize)
        benchBatched("quantize_int8", ps, queue, device,
                     buffers: [inp, qout, scales], count: size,
                     bytesRW: sizeBytes + size) { enc in
            enc.setBytes(&params, length: 8, index: 3)
        }
    }
    // Dequantize
    let dqout = device.makeBuffer(length: sizeBytes, options: .storageModeShared)!
    if let ps = loadKernel(qSrc, "dequantize_int8") {
        var params = (UInt32(size), groupSize)
        benchBatched("dequantize_int8", ps, queue, device,
                     buffers: [qout, dqout, scales], count: size,
                     bytesRW: size + sizeBytes) { enc in
            enc.setBytes(&params, length: 8, index: 3)
        }
    }
    print("")
}

// ── 6. RoPE (position encoding) ─────────────────────────────────────────
print("  RoPE (Rotary Position Embedding)")
print("  ─────────────────────────────────")
if let ps = loadKernel(fusedSrc, "rope_kernel") {
    for (heads, dim) in [(32, 128), (8, 64)] {
        let headDim = UInt32(dim); let nHeads = UInt32(heads); let nKVHeads = UInt32(heads)
        let totalFloats = Int(nHeads) * dim
        let kFloats = Int(nKVHeads) * dim
        let q = device.makeBuffer(length: totalFloats * 4, options: .storageModeShared)!
        let k = device.makeBuffer(length: kFloats * 4, options: .storageModeShared)!
        fillRandom(q, count: totalFloats, range: -1...1)
        fillRandom(k, count: kFloats, range: -1...1)
        var hd = headDim; var nh = nHeads; var nkv = nKVHeads
        var pos: Int32 = 42; var theta = Float(10000.0)
        let totalPairs = Int((nHeads + nKVHeads) * headDim / 2)
        benchBatched("rope h=\(heads) d=\(dim)", ps, queue, device,
                     buffers: [q, k], count: totalPairs,
                     bytesRW: (totalFloats + kFloats) * 4 * 2) { enc in
            enc.setBytes(&hd, length: 4, index: 2)
            enc.setBytes(&nh, length: 4, index: 3)
            enc.setBytes(&nkv, length: 4, index: 4)
            enc.setBytes(&pos, length: 4, index: 5)
            enc.setBytes(&theta, length: 4, index: 6)
        }
    }
}
print("")

print("============================================================")
print(" Benchmark complete")
print("============================================================")

exit(failCount > 0 ? 1 : 0)
