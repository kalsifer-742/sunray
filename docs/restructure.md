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

Net error count vs baseline: 14 → 9 (the 5 sampler-cascade errors are gone via a temporary Sampler::inner() shim; the remaining 9 are all pre-existing WIP in graph.rs, scene.rs, and unrelated [f32;3] mismatches in lib.rs).

What's in place
- src/shader_compiler/{mod.rs, compiler.rs} — ShaderCompiler owns a slang::GlobalSession, enables spvDescriptorHeapEXT, targets spirv_1_5. compile(module_name, entry_point) -> Vec<u8>.
- shaders/postprocess.slang — uses DescriptorHandle<RWTexture2D<float4>> for input/output, [vk::push_constant] ConstantBuffer<PostprocessPC>.
- compute_pipeline.rs::ComputePipeline::new_heap — pipeline layout with no descriptor sets, PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT on the compute pipeline itself, push-constant range only.
- PostprocessPushConstant is now { input_idx: u32, output_idx: u32, exposure: f32 } — the two indices match Slang's DescriptorHandle lowering.
- lib.rs::cmd_postprocess_image — drops cmd_bind_descriptor_sets; calls core.descriptor_heap().cmd_bind(cmd_buf) before dispatch; push-constants input_image.storage_slot() / output_image.storage_slot().
- Sampler::inner() shim restored (lazy vkCreateSampler, destroyed in Drop) so the still-legacy passes (raygen/denoise/temporal) keep compiling. Marked TEMPORARY in code, removed once they migrate.

To actually run the renderer end-to-end you need to fix the 9 pre-existing errors (graph.rs render_graph WIP, scene.rs:223, lib.rs:676/1677). Those are outside the heap migration's scope.

Next, when you're ready to test:
- Run; the only pass actually exercising VK_EXT_descriptor_heap is postprocess. Other passes still use descriptor sets.
- Validation layer + GPUAV will tell you fast if the heap binding or shader index math is wrong.
- If postprocess shows the tonemapped image correctly: pattern is proven; we can roll the same approach into temporal → denoise → raygen, then delete descriptor_sets/ and the Sampler::inner() shim.