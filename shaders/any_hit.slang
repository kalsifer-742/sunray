// Any-hit: alpha-cutout test for glTF MASK materials. Geometry that is fully
// opaque (alpha_mode == 0) returns immediately so the closest-hit can run.
// Mirrors shaders/any_hit.glsl.

import rt_types;
import rt_utils;

[vk::push_constant] ConstantBuffer<RaytracingPC> pc;

[shader("anyhit")]
void any_hit(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    StructuredBuffer<MeshInfo> meshes = pc.meshes_info;
    StructuredBuffer<TextureEntry> textures_lookup = pc.textures_lookup;

    uint blas_instance_id = InstanceID();
    MeshInfo mesh_info = meshes[blas_instance_id];
    Material material = mesh_info.material;

    // OPAQUE — closest-hit handles it.
    if (material.alpha_mode == 0) {
        return;
    }

    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x,
                         attribs.barycentrics.y);

    uint index_offset = PrimitiveIndex() * 3;
    uint i0 = mesh_info.indices[index_offset + 0];
    uint i1 = mesh_info.indices[index_offset + 1];
    uint i2 = mesh_info.indices[index_offset + 2];

    float2 uv0 = mesh_info.vertices[i0].base_color_tex_coord;
    float2 uv1 = mesh_info.vertices[i1].base_color_tex_coord;
    float2 uv2 = mesh_info.vertices[i2].base_color_tex_coord;
    float2 uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    float4 base_color = sample_texture(textures_lookup, material.base_color_texture_index, uv, material.base_color_value);

    if (base_color.a < material.alpha_cutoff) {
        IgnoreHit();
    }
}
