slang 
dlss/fsr/xess ecc... divisi correttamente  
ptlas e clustered blas o blas compressi 
(BLAS compaction - can free up to 50% device memory usage
Face culling
Grouping together objects that are on the same axis)
restir pt + advancement 
antilag Nvidia ecc... 
sectioning della pipeline ,magari ispirata da bevy(dag), con possibili diversi attacchi, renderebbe molto più semplice l'integrazione 
HDR 
GLTF supporto avanzato : animazioni,luci e   
Post processing 
Api più stabili,cambiando visibilità e struttura  
Blender teoricamente possibile tramite minimizzazione di python usando numpy ecc...
Documentazione
Criterion per micro benchmarking e benchmarking tools testing ecc...

Solutions 
Render graph DAG  
Builder pattern 
Opaque types
Facade pattern(spillitng public e private in 2 parti frontend e backend ecc)
Data Oriented Design Pattern con Structure of Arrays
Ecs meglio strutturata
RenderPass trait e dipendenze di buffer con rendergraph 
Pipeline caching
Slang che però richiede il compilatore di shader,che va fatto girare aot o jit
Strategy Patter per fsr/dlss/ 
Egui invece di bevy_ui

Step 4 (postprocess slice) — done.


● All 5 raytracing stages now compile from Slang to SPIR-V at build time. Stopping here for this turn to keep the change reviewable.

Status:
- ✅  rt_types.slang + rt_utils.slang — shared structs, helpers, ReSTIR types, BRDF, pack/unpack helpers, heap-indirected texture sampling
- ✅  ray_miss.slang, any_hit.slang, closest_hit.slang, ray_gen_ris.slang, ray_gen_final.slang — all five RT entry points
- ✅  build.rs compiles them all with matrix_layout_row(false) to match nalgebra's column-major matrices

Things that needed working around (worth knowing):
- Slang's vk::BufferPointer<T,A> has no + overload, no .get(), no member forwarding — switched all BLAS BDA pointers to bare T* (Slang's Ptr<T> with __subscript)
- Slang's unpackUnorm4x8, unpackHalf2x16, unpackSnorm2x16 (and a few others) are "undefined identifier" on this SDK — inlined manual versions in rt_utils.slang
- DescriptorHandle<ConstantBuffer<T>> doesn't forward .field access — matrices now go through DescriptorHandle<StructuredBuffer<Matrices>> and are read as pc.matrices[0].view_inverse (CPU side will need Buffer::storage_slot() plus STORAGE_BUFFER usage when we get to CPU integration)
- ResourceDescriptorHeap / SamplerDescriptorHeap (HLSL SM6.6) aren't recognised either — texture indirection now stores typed DescriptorHandle<Texture2D> / DescriptorHandle<SamplerState> pairs in the lookup buffer (same 16-byte stride: two uint2s)

Next up (separate turn): heap-mode RayTracingPipeline constructor + texture-lookup buffer on resource_manager + switching cmd_raytracing_render to push the new push constant. Tasks #5/#6/#7 are still pending.