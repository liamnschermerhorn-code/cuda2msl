#!/bin/bash
cd "$(dirname "$0")/../.."
swift tests/metal/validate_kernels.swift src/shaders/ --all 2>&1 | tee tests/metal/validation_results.txt
