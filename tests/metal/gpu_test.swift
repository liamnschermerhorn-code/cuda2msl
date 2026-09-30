#!/usr/bin/env swift
// gpu_test.swift
// Stage 2: Load every .metal file, create a pipeline state for every kernel
// function, then run numerical validation on kernels we can identify by name.
//
// Usage:
//   swift gpu_test.swift /path/to/metal/files
//   swift gpu_test.swift .

import Metal
import Foundation

// ── Helpers ─────────────────────────────────────────────────────────────────

func fail(_ msg: String) -> Never { fputs("FATAL: \(msg)\n", stderr); exit(1) }

var passCount = 0
var failCount = 0

func check(_ cond: Bool, _ label: String, detail: String = "") {
    if cond {
        print("  ✓  \(label)")
        passCount += 1
    } else {
        print("  ✗  \(label)\(detail.isEmpty ? "" : "  [\(detail)]")")
        failCount += 1
    }
}

// Fill a buffer with sequential floats: 1,2,3,...
func fillSeq(_ buf: MTLBuffer, count: Int, start: Float = 1) {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = start + Float(i) }
}

// Fill with a constant
func fillConst(_ buf: MTLBuffer, count: Int, value: Float) {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = value }
}

// Read floats back
func readFloats(_ buf: MTLBuffer, count: Int) -> [Float] {
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

// Run a 1D kernel over `count` elements with given buffers
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

// ── Main ─────────────────────────────────────────────────────────────────────

let args = CommandLine.arguments
let searchDir = args.count > 1 ? args[1] : "."

guard let device = MTLCreateSystemDefaultDevice() else { fail("No Metal device") }
guard let queue  = device.makeCommandQueue()       else { fail("No command queue") }

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(" GPU runtime test  →  \(searchDir)")
print(" Device: \(device.name)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

let fm = FileManager.default
guard let enumerator = fm.enumerator(atPath: searchDir) else { fail("Cannot read directory") }

var metalFiles: [String] = []
while let item = enumerator.nextObject() as? String {
    if item.hasSuffix(".metal") { metalFiles.append((searchDir as NSString).appendingPathComponent(item)) }
}
metalFiles.sort()

if metalFiles.isEmpty { fail("No .metal files found in \(searchDir)") }

// ── Per-file tests ────────────────────────────────────────────────────────────

for path in metalFiles {
    let name = (path as NSString).lastPathComponent
    print("\n── \(name)")

    // 1. Compile the source into a library
    let src: String
    do { src = try String(contentsOfFile: path, encoding: .utf8) }
    catch { print("  ✗  cannot read file: \(error)"); failCount += 1; continue }

    let opts = MTLCompileOptions()
    opts.languageVersion = .version3_0
    let lib: MTLLibrary
    do { lib = try device.makeLibrary(source: src, options: opts) }
    catch let e as NSError {
        print("  ✗  compile failed:")
        let msg = e.localizedDescription
        for line in msg.split(separator: "\n") { print("     \(line)") }
        failCount += 1; continue
    }
    check(true, "compiles on GPU driver")

    // 2. Create pipeline states for every kernel function
    let funcNames = lib.functionNames
    check(!funcNames.isEmpty, "has at least one function (found \(funcNames.count))")

    var pipelines: [String: MTLComputePipelineState] = [:]
    for fname in funcNames {
        guard let fn = lib.makeFunction(name: fname) else {
            check(false, "makeFunction(\(fname))"); continue
        }
        do {
            let ps = try device.makeComputePipelineState(function: fn)
            pipelines[fname] = ps
            check(true, "pipeline: \(fname)  [max threads/group: \(ps.maxTotalThreadsPerThreadgroup)]")
        } catch {
            check(false, "pipeline: \(fname)", detail: error.localizedDescription)
        }
    }

    // 3. Numerical tests — matched by function name patterns
    let N = 1024
    let bytes = N * MemoryLayout<Float>.stride

    for (fname, ps) in pipelines {
        let lo = fname.lowercased()

        // ── elementwise add / vadd / vector_add ──────────────────────────────
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
            check(allCorrect, "numerical: \(fname)(3,5) = 8  [\(out[0])]")
        }

        // ── elementwise multiply / scale ─────────────────────────────────────
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
            check(allCorrect, "numerical: \(fname)(4, scale=2.5) = 10  [\(out[0])]")
        }

        // ── ReLU ─────────────────────────────────────────────────────────────
        if lo.contains("relu") {
            let inp = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let out = device.makeBuffer(length: bytes, options: .storageModeShared)!
            let ptr = inp.contents().bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { ptr[i] = Float(i % 2 == 0 ? i : -i) }  // alternating +/-
            var n = Int32(N)
            dispatch1D(pipeline: ps, queue: queue, buffers: [inp, out], count: N) { enc in
                enc.setBytes(&n, length: 4, index: 2)
            }
            let result = readFloats(out, count: N)
            let allCorrect = (0..<N).allSatisfy { i in
                result[i] >= 0.0 && (result[i] == Float(i % 2 == 0 ? i : 0))
            }
            check(allCorrect, "numerical: \(fname) — all outputs ≥ 0  [\(result[0]), \(result[1])]")
        }

        // ── GELU ─────────────────────────────────────────────────────────────
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
            // GELU(x) ≈ 0 for large negative x, ≈ x for large positive x
            let bigNeg = result[0]    // GELU(very negative) ≈ 0
            let bigPos = result[N-1]  // GELU(very positive) ≈ x
            check(abs(bigNeg) < 0.01, "numerical: \(fname) GELU(very neg) ≈ 0  [\(bigNeg)]")
            check(bigPos > 0, "numerical: \(fname) GELU(very pos) > 0  [\(bigPos)]")
        }

        // ── softmax ──────────────────────────────────────────────────────────
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
            check(abs(sum - 1.0) < 0.01, "numerical: \(fname) — outputs sum to 1.0  [sum=\(sum)]")
            check(result.allSatisfy { $0 > 0 }, "numerical: \(fname) — all outputs > 0")
        }

        // ── layer norm ───────────────────────────────────────────────────────
        if lo.contains("layernorm") || lo.contains("layer_norm") {
            let inp  = device.makeBuffer(length: bytes, options: .storageModeShared)!
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
            check(abs(mean) < 0.1, "numerical: \(fname) — output mean ≈ 0  [mean=\(mean)]")
        }

        // ── copy / memcpy ────────────────────────────────────────────────────
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
            check(correct, "numerical: \(fname) — dst matches src  [\(result[0])]")
        }
    }
}

// ── Summary ──────────────────────────────────────────────────────────────────
print("")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(" \(passCount) passed  ·  \(failCount) failed")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
exit(failCount > 0 ? 1 : 0)
