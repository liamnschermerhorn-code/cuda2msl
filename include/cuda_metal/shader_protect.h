#pragma once

// Shader Binary Obfuscation
//
// M5+ tier: .metallib files are encrypted at build time and decrypted at load time.
//   - AES-256-CTR encryption via CommonCrypto
//   - Key derived from device hardware UUID + embedded salt
//   - Symbol names stripped and remapped
//   - Prevents reverse engineering of Metal shader IP
//
// Legacy tier: standard .metallib loading (no obfuscation).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuda_metal {

// File header for encrypted .metallib
struct ProtectedShaderHeader {
    uint8_t  magic[4];    // "CMSP" (CUDA-Metal Shader Protected)
    uint32_t version;     // 1
    uint32_t flags;       // bit 0: encrypted, bit 1: symbols stripped
    uint32_t key_salt_len;
    uint32_t iv_len;
    uint32_t payload_len; // original .metallib size
    uint32_t checksum;    // CRC32 of original payload
    // Followed by: [salt bytes] [iv bytes] [encrypted payload]
};

// Encrypt a .metallib binary into a protected format.
// Returns the encrypted buffer, or empty on failure.
std::vector<uint8_t> protect_shader(const std::vector<uint8_t>& metallib_data,
                                     const std::string& passphrase);

// Encrypt a .metallib file and write to output path.
bool protect_shader_file(const std::string& input_path,
                          const std::string& output_path,
                          const std::string& passphrase);

// Decrypt a protected shader buffer back to raw .metallib.
// Uses device hardware UUID if passphrase is empty.
std::vector<uint8_t> unprotect_shader(const std::vector<uint8_t>& protected_data,
                                       const std::string& passphrase = "");

// Decrypt from file.
std::vector<uint8_t> unprotect_shader_file(const std::string& path,
                                            const std::string& passphrase = "");

// Get the device hardware UUID (for key derivation).
std::string get_hardware_uuid();

// Compute CRC32 of data.
uint32_t crc32(const void* data, size_t len);

} // namespace cuda_metal
