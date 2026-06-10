// Shared types for the Slang/heap-mode raytracing pipeline. Layouts mirror
// the Rust `#[repr(C)]` structs and the GLSL std430 buffers exactly so the
// same backing buffers can be read from either code path during the
// GLSL → Slang transition.

module rt_types;

// --- Per-ray payload --------------------------------------------------------
public struct RayPayload {
    public float3 emission;
    public float dist;
    public uint albedo_packed;
    public uint normal_packed;
    public uint material_info;
    public uint transmission_ior_packed;
}

// --- Vertex / mesh ----------------------------------------------------------
// Explicit padding fields mirror the `#[repr(C, packed)]` Rust `Vertex` in
// `src/vulkan_abstraction/gltf/vertex.rs`. Slang's `T*` (used for BDA mesh
// access) lays the struct out in scalar layout — without the explicit pad
// fields, `normal` would land at offset 12 instead of 16, breaking every
// closest-hit / any-hit read. 96 bytes total.
public struct VertexAttributes {
    public float3 position;
    public float _pad0;
    public float3 normal;
    public float _pad1;
    public float4 tangent;
    public float2 base_color_tex_coord;
    public float2 metallic_roughness_tex_coord;
    public float2 normal_tex_coord;
    public float2 occlusion_tex_coord;
    public float2 emissive_tex_coord;
    public float2 _pad3;
}

// BLAS vertex/index buffers are reached via buffer-device-address pointers
// (one address per BLAS), not heap descriptors — the heap would otherwise
// need a slot per BLAS buffer. Slang's `T*` lowers to a SPIR-V
// `PhysicalStorageBuffer` pointer and supports subscript via the stdlib's
// `Ptr<T>::__subscript`, which is the equivalent of GLSL's
// `buffer_reference { T v[]; }` unbounded-array idiom.
//
// **Material fields are inlined** rather than nested as a `Material` struct:
// Slang's heap-mode (`OpBufferPointerEXT`) access chain for a nested struct
// member drops the parent member offset, so every nested-material read would
// land 16 bytes off. Field order/sizes mirror `#[repr(C, packed)]` `Material`
// in `src/vulkan_abstraction/resources/material.rs` exactly (112 bytes after
// the two pointers).
// TODO issue the problem to the slang repo and look at alternatives and alignment
//
// MeshInfo is read through a `StructuredBuffer` → std430 layout: every float4
// must sit at a 16-aligned offset (hence the `_material_pad_mid_*` pair) and
// the struct size must be a 16-multiple (hence the trailing pads) so the
// std430 array stride (128) matches the CPU side.
//
// `material_*_image` / `material_*_sampler` are *resolved descriptor heap
// slots* (filled in at scene-load time on the CPU) — there is no texture
// indirection buffer. `NULL_TEXTURE` in an image slot means "no texture".
public struct MeshInfo {
    public VertexAttributes* vertices;                          // 0
    public uint* indices;                                       // 8
    public float4 material_base_color_value;                    // 16
    public float material_metallic_factor;                      // 32
    public float material_roughness_factor;                     // 36
    public float _material_pad_mid_0;                           // 40
    public float _material_pad_mid_1;                           // 44
    public float4 material_emissive_factor;                     // 48
    public uint material_alpha_mode;                            // 64
    public float material_alpha_cutoff;                         // 68
    public float material_transmission_factor;                  // 72
    public float material_ior;                                  // 76
    public uint material_base_color_image;                      // 80
    public uint material_base_color_sampler;                    // 84
    public uint material_metallic_roughness_image;              // 88
    public uint material_metallic_roughness_sampler;            // 92
    public uint material_normal_image;                          // 96
    public uint material_normal_sampler;                        // 100
    public uint material_occlusion_image;                       // 104
    public uint material_occlusion_sampler;                     // 108
    public uint material_emissive_image;                        // 112
    public uint material_emissive_sampler;                      // 116
    public uint _material_pad_end_0;                            // 120
    public uint _material_pad_end_1;                            // 124
}

// --- Emissive lighting ------------------------------------------------------
public struct EmissiveTriangle {
    public float4 v0;
    public float4 v1;
    public float4 v2;
    public float4 emission;
}

public struct EmissiveIndirectionEntry {
    public uint blas_tri_index;
    public uint entity_id;
}

public struct EntityTransform {
    public float4 rows[3];
}

// --- Camera matrices --------------------------------------------------------
// Stored as 16 flat `float4` columns instead of 4 `float4x4` (or even 4
// `float4[4]`) members because Slang's lowering of struct-member access
// through `OpUntypedPointerKHR` (descriptor-heap path) emits an OpAccessChain
// with a mismatched storage class (Function result over StorageBuffer base)
// for any composite-typed member — matrices, arrays, sub-structs — producing
// invalid SPIR-V → DEVICE_LOST. Scalar-typed members (including `float4`) go
// through the typed-pointer path and are fine. The on-disk byte layout is
// identical (column-major, 16 floats per matrix = 64 bytes) so the Rust upload
// of `MatricesBufferContents` (4 × nalgebra::Matrix4 column-major) is unchanged.
public struct Matrices {
    public float4 vi0; public float4 vi1; public float4 vi2; public float4 vi3;
    public float4 pi0; public float4 pi1; public float4 pi2; public float4 pi3;
    public float4 vp0; public float4 vp1; public float4 vp2; public float4 vp3;
    public float4 pp0; public float4 pp1; public float4 pp2; public float4 pp3;
}

// --- ReSTIR reservoirs ------------------------------------------------------
public struct Reservoir {
    public float3 light_pos;
    public float w_sum;
    public float3 light_normal;
    public float M;
    public uint light_idx;
    public float W;
    public uint hit_normal_packed;
    public float depth;
}

public struct ReservoirGI {
    public float3 sample_pos;
    public float w_sum;
    public float3 sample_radiance;
    public float M;
    public uint sample_normal_packed;
    public float W;
    public uint hit_normal_packed;
    public float depth;
}

// --- Push constant ----------------------------------------------------------
// Every binding from the old `RaytracingDescriptorSetLayout` is now a
// `DescriptorHandle<T>` field here. `reservoirs[2]` and `reservoirs_gi[2]`
// keep the same A/B ping-pong scheme, indexed by `frame_count & 1u`.
// Texture access goes through the heap slots stored per material in
// `MeshInfo` — there is no texture indirection buffer.
public struct
RaytracingPC {
    // BDA-like uint64 instead of a DescriptorHandle<AS> — see
    // `rt_utils.slang::tlas_from_address` for the workaround rationale
    // (Slang issue #10671). Same 8-byte footprint, so the rest of the PC
    // layout is unchanged.
    public uint64_t tlas;
    public DescriptorHandle<RWTexture2D<float4>> raw_color;
    public DescriptorHandle<RWTexture2D<float>>  depth_img;
    public DescriptorHandle<RWTexture2D<float4>> normal_img;
    public DescriptorHandle<RWTexture2D<float4>> diffuse_img;
    public DescriptorHandle<RWTexture2D<float2>> motion_vec_img;

    // BDA pointer instead of a `DescriptorHandle<StructuredBuffer<Matrices>>` —
    // Slang's heap-descriptor lowering miscomputes the storage class when
    // accessing struct members of a `StructuredBuffer<T>` element, producing
    // invalid SPIR-V. The `T*` (PhysicalStorageBuffer) path doesn't have that
    // bug. The CPU push-constant field stays 8 bytes (a `uint64_t` device
    // address instead of a `uint2` heap handle).
    public Matrices* matrices;
    public DescriptorHandle<StructuredBuffer<MeshInfo>> meshes_info;
    public DescriptorHandle<StructuredBuffer<EmissiveTriangle>> emissive_triangles;
    public DescriptorHandle<StructuredBuffer<EmissiveIndirectionEntry>> emissive_indirection;
    public DescriptorHandle<StructuredBuffer<EntityTransform>> entity_transforms;

    public DescriptorHandle<Texture2D<float4>> blue_noise_tex;
    public DescriptorHandle<SamplerState> blue_noise_sampler;

    // BDA pointers instead of `DescriptorHandle<RWStructuredBuffer<T>>` — Slang
    // decomposes whole-struct writes into per-field stores with mismatched
    // storage classes when the buffer is reached through the heap descriptor
    // path, producing invalid SPIR-V. `Reservoir*` (PhysicalStorageBuffer)
    // doesn't have that bug. Each pointer is 8 bytes, matching the previous
    // `DescriptorHandle` lowering, so the rest of the PC layout is unchanged.
    public Reservoir* reservoirs[2];
    public ReservoirGI* reservoirs_gi[2];

    public uint frame_count;
    public uint use_srgb;
}

public static const uint NULL_TEXTURE = 0xFFFFFFFFu;
