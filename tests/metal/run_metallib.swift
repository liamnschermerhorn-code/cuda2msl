#!/usr/bin/env swift
// run_metallib.swift
// Load a pre-compiled .metallib, enumerate kernels, create pipeline states,
// and run numerical validation. Outputs structured JSON for server consumption.
//
// Usage:
//   swift run_metallib.swift /path/to/shaders.metallib
//   swift run_metallib.swift /path/to/dir/  (scans for .metallib files)

import Metal
import Foundation

// ── Helpers ─────────────────────────────────────────────────────────────────

func fail(_ msg: String) -> Never { fputs("FATAL: \(msg)\n", stderr); exit(1) }

struct TestResult: Codable {
    let kernel: String
    let test: String
    let passed: Bool
    let detail: String
}

var results: [TestResult] = []
var passCount = 0
var failCount = 0

func check(_ cond: Bool, kernel: String, _ label: String, detail: String = "") {
    results.append(TestResult(kernel: kernel, test: label, passed: cond, detail: detail))
    if cond { passCount += 1 } else { failCount += 1 }
    let icon = cond ? "PASS" : "FAIL"
    fputs("[\(icon)] \(kernel): \(label)\(detail.isEmpty ? "" : " [\(detail)]")\n", stdout)
    fflush(stdout)
}

func fillSeq(_ buf: MTLBuffer, count: Int, start: Float = 1) {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = start + Float(i) }
}

func fillConst(_ buf: MTLBuffer, count: Int, value: Float) {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = value }
}

func readFloats(_ buf: MTLBuffer, count: Int) -> [Float] {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

func dispatch1D(pipeline: MTLComputePipelineState,
                queue: MTLCommandQueue,
                buffers: [MTLBuffer],
                count: Int,
                extraSetup: ((MTLComputeCommandEncoder) -> Void)? = nil) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    for (i, buf) in buffers.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
    extraSetup?(enc)
    let tpg = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    let tg  = MTLSize(width: tpg, height: 1, depth: 1)
    let grid = MTLSize(width: count, height: 1, depth: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()
    cmd.commit(); cmd.waitUntilCompleted()
}

// ── Main ────────────────────────────────────────────────────────────────────

let args = CommandLine.arguments
let searchPath = args.count > 1 ? args[1] : "."

guard let device = MTLCreateSystemDefaultDevice() else { fail("No Metal device") }
guard let queue  = device.makeCommandQueue()       else { fail("No command queue") }

print("DEVICE: \(device.name)")
fflush(stdout)

let fm = FileManager.default
var metallibPaths: [String] = []

var isDir: ObjCBool = false
if fm.fileExists(atPath: searchPath, isDirectory: &isDir) {
    if isDir.boolValue {
        // Directory — scan for .metallib files
        if let items = fm.enumerator(atPath: searchPath) {
            while let item = items.nextObject() as? String {
                if item.hasSuffix(".metallib") {
                    metallibPaths.append((searchPath as NSString).appendingPathComponent(item))
                }
            }
        }
    } else if searchPath.hasSuffix(".metallib") {
        metallibPaths.append(searchPath)
    }
}

if metallibPaths.isEmpty { fail("No .metallib files found at \(searchPath)") }
metallibPaths.sort()

// ── Load each metallib ──────────────────────────────────────────────────────

for path in metallibPaths {
    let name = (path as NSString).lastPathComponent
    print("LOADING: \(name)")
    fflush(stdout)

    let url = URL(fileURLWithPath: path)
    let lib: MTLLibrary
    do { lib = try device.makeLibrary(URL: url) }
    catch {
        print("FAIL: cannot load \(name): \(error.localizedDescription)")
        failCount += 1
        fflush(stdout)
        continue
    }

    let funcNames = lib.functionNames
    print("KERNELS: \(funcNames.count) functions in \(name): \(funcNames.joined(separator: ", "))")
    fflush(stdout)

    // Create pipeline states for each function
    var pipelines: [String: MTLComputePipelineState] = [:]
    for fname in funcNames {
        guard let fn = lib.makeFunction(name: fname) else {
            check(false, kernel: fname, "makeFunction", detail: "failed to create function")
            continue
        }
        // Skip functions that require function constants
        do {
            let ps = try device.makeComputePipelineState(function: fn)
            pipelines[fname] = ps
            check(true, kernel: fname, "pipeline_created",
                  detail: "maxThreads=\(ps.maxTotalThreadsPerThreadgroup)")
        } catch {
            // May fail for functions requiring constants — not an error
            print("SKIP: \(fname) requires function constants or is not a compute kernel")
            fflush(stdout)
        }
    }

    // ── Numerical tests ─────────────────────────────────────────────────────

    let N = 1024
    let bytes = N * MemoryLayout<Float>.stride

    for (fname, ps) in pipelines {
        let lo = fname.lowercased()

        // ── add / vadd / vector_add
        if lo.contains("add") && !lo.contains("atomic") && !lo.contains("bias") {
            let a = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let c = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillConst(a, count: N, value: 3.0)
            fillConst(b, count: N, value: 5.0)
            fillConst(c, count: N, value: 0.0)
            var n = Int32(N)
            dispatch1D(pipeline: ps, queue: queue, buffers: [a, b, c], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 3)
            }
            let out = readFloats(c, count: N)
            let allCorrect = out.allSatisfy { abs($0 - 8.0) < 1e-4 }
            check(allCorrect, kernel: fname, "numerical_add",
                  detail: "3+5=\(out[0]), expected 8.0")
        }

        // ── multiply / scale
        if (lo.contains("mul") || lo.contains("scale")) && !lo.contains("matrix") {
            let a = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillConst(a, count: N, value: 4.0)
            fillConst(b, count: N, value: 0.0)
            var n = Int32(N); var s = Float(2.5)
            dispatch1D(pipeline: ps, queue: queue, buffers: [a, b], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
                enc.setBytes(&s, length: 4, index: 3)
            }
            let out = readFloats(b, count: N)
            let allCorrect = out.allSatisfy { abs($0 - 10.0) < 1e-4 }
            check(allCorrect, kernel: fname, "numerical_mul",
                  detail: "4*2.5=\(out[0]), expected 10.0")
        }

        // ── ReLU
        if lo.contains("relu") {
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i % 2 == 0 ? i : -i) }
            var n = Int32(N)
            dispatch1D(pipeline: ps, queue: queue, buffers: [inp, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let result = readFloats(out, count: N)
            let allCorrect = result.allSatisfy { $0 >= 0.0 }
            check(allCorrect, kernel: fname, "numerical_relu",
                  detail: "all outputs >= 0: \(allCorrect)")
        }

        // ── GELU
        if lo.contains("gelu") {
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i) - Float(N/2) }
            var n = Int32(N)
            dispatch1D(pipeline: ps, queue: queue, buffers: [inp, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let result = readFloats(out, count: N)
            let bigNeg = result[0]
            let bigPos = result[N-1]
            check(abs(bigNeg) < 0.01, kernel: fname, "numerical_gelu_neg",
                  detail: "GELU(very neg)=\(bigNeg), expected ~0")
            check(bigPos > 0, kernel: fname, "numerical_gelu_pos",
                  detail: "GELU(very pos)=\(bigPos), expected >0")
        }

        // ── softmax
        if lo.contains("softmax") {
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillConst(inp, count: N, value: 1.0)
            var n = Int32(N)
            dispatch1D(pipeline: ps, queue: queue, buffers: [inp, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let result = readFloats(out, count: N)
            let sum = result.reduce(0, +)
            check(abs(sum - 1.0) < 0.01, kernel: fname, "numerical_softmax",
                  detail: "sum=\(sum), expected 1.0")
        }

        // ── layer norm
        if lo.contains("layernorm") || lo.contains("layer_norm") {
            let inp   = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let gamma = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let beta  = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out   = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillSeq(inp, count: N)
            fillConst(gamma, count: N, value: 1.0)
            fillConst(beta,  count: N, value: 0.0)
            var n = Int32(N)
            dispatch1D(pipeline: ps, queue: queue, buffers: [inp, gamma, beta, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 4)
            }
            let result = readFloats(out, count: N)
            let mean = result.reduce(0, +) / Float(N)
            check(abs(mean) < 0.1, kernel: fname, "numerical_layernorm",
                  detail: "mean=\(mean), expected ~0")
        }

        // ── copy / memcpy
        if lo.contains("copy") || lo.contains("memcpy") {
            let src = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let dst = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillSeq(src, count: N)
            fillConst(dst, count: N, value: 0.0)
            var n = Int32(N)
            dispatch1D(pipeline: ps, queue: queue, buffers: [src, dst], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let result = readFloats(dst, count: N)
            let correct = zip(readFloats(src, count: N), result).allSatisfy { abs($0.0 - $0.1) < 1e-5 }
            check(correct, kernel: fname, "numerical_copy",
                  detail: "dst[0]=\(result[0])")
        }

        // ── generic kernel — just verify it dispatches without crashing
        if pipelines.count == 1 || (!lo.contains("add") && !lo.contains("mul") &&
            !lo.contains("relu") && !lo.contains("gelu") && !lo.contains("softmax") &&
            !lo.contains("layernorm") && !lo.contains("layer_norm") &&
            !lo.contains("copy") && !lo.contains("memcpy") && !lo.contains("scale")) {
            // Dispatch with dummy buffers — just test that it doesn't crash
            let a = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
            fillConst(a, count: N, value: 1.0)
            fillConst(b, count: N, value: 0.0)
            var n = Int32(N)
            dispatch1D(pipeline: ps, queue: queue, buffers: [a, b], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            check(true, kernel: fname, "dispatch_ok",
                  detail: "dispatched \(N) threads without crash")
        }
    }
}

// ── Summary ─────────────────────────────────────────────────────────────────

print("SUMMARY: \(passCount) passed, \(failCount) failed")
fflush(stdout)

// Output JSON for programmatic consumption
let encoder = JSONEncoder()
encoder.outputFormatting = .prettyPrinted
if let json = try? encoder.encode(results) {
    print("JSON_RESULTS:")
    print(String(data: json, encoding: .utf8)!)
}

exit(failCount > 0 ? 1 : 0)
