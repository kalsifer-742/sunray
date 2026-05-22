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
public struct VertexAttributes {
    public float3 position;
    public float3 normal;
    public float4 tangent;
    public float2 base_color_tex_coord;
    public float2 metallic_roughness_tex_coord;
    public float2 normal_tex_coord;
    public float2 occlusion_tex_coord;
    public float2 emissive_tex_coord;
}

public struct Material {
    public float4 base_color_value;
    public uint base_color_texture_index;

    public float metallic_factor;
    public float roughness_factor;
    public uint metallic_roughness_texture_index;

    public uint normal_texture_index;
    public uint occlusion_texture_index;

    public float4 emissive_factor;
    public uint emissive_texture_index;

    public uint alpha_mode;
    public float alpha_cutoff;

    public float transmission_factor;
    public float ior;

    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

// BLAS vertex/index buffers are reached via buffer-device-address pointers
// (one address per BLAS), not heap descriptors — the heap would otherwise
// need a slot per BLAS buffer. Slang's `T*` lowers to a SPIR-V
// `PhysicalStorageBuffer` pointer and supports subscript via the stdlib's
// `Ptr<T>::__subscript`, which is the equivalent of GLSL's
// `buffer_reference { T v[]; }` unbounded-array idiom.
public struct MeshInfo {
    public VertexAttributes* vertices;
    public uint* indices;
    public Material material;
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
public struct Matrices {
    public float4x4 view_inverse;
    public float4x4 proj_inverse;
    public float4x4 view_proj;
    public float4x4 prev_view_proj;
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
// `textures_lookup` replaces the old 1024-element combined-sampler array:
// each entry is `(image_heap_index, sampler_heap_index)` so material indices
// (`base_color_texture_index`, …) keep their original meaning.
public struct RaytracingPC {
    public DescriptorHandle<RaytracingAccelerationStructure> tlas;
    public DescriptorHandle<RWTexture2D<float4>> raw_color;
    public DescriptorHandle<RWTexture2D<float>>  depth_img;
    public DescriptorHandle<RWTexture2D<float4>> normal_img;
    public DescriptorHandle<RWTexture2D<float4>> diffuse_img;
    public DescriptorHandle<RWTexture2D<float2>> motion_vec_img;

    // Slang doesn't forward `.field` access through `DescriptorHandle<ConstantBuffer<T>>`,
    // so the matrices live in a single-element `StructuredBuffer<Matrices>` instead. The
    // descriptor is a storage buffer on the CPU side (we use `Buffer::storage_slot()`).
    public DescriptorHandle<StructuredBuffer<Matrices>> matrices;
    public DescriptorHandle<StructuredBuffer<MeshInfo>> meshes_info;
    public DescriptorHandle<StructuredBuffer<EmissiveTriangle>> emissive_triangles;
    public DescriptorHandle<StructuredBuffer<EmissiveIndirectionEntry>> emissive_indirection;
    public DescriptorHandle<StructuredBuffer<EntityTransform>> entity_transforms;

    public DescriptorHandle<Texture2D<float4>> blue_noise_tex;
    public DescriptorHandle<SamplerState> blue_noise_sampler;

    public DescriptorHandle<RWStructuredBuffer<Reservoir>> reservoirs[2];
    public DescriptorHandle<RWStructuredBuffer<ReservoirGI>> reservoirs_gi[2];

    public DescriptorHandle<StructuredBuffer<TextureEntry>> textures_lookup;

    public uint frame_count;
    public uint use_srgb;
}

// Per-texture indirection entry. Two `DescriptorHandle` fields each lower
// to a `uint2`, so the CPU writes 4 × u32 per entry: `(image_heap_idx, 0,
// sampler_heap_idx, 0)`. The pad words are required because Slang's
// `DescriptorHandle<T>` is two 32-bit components even though we only use
// the low one.
public struct TextureEntry {
    public DescriptorHandle<Texture2D<float4>> image;
    public DescriptorHandle<SamplerState> sampler;
}

public static const uint NULL_TEXTURE = 0xFFFFFFFFu;
