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


3. Pre-existing logic bug (not from this refactor, but flagged)

src/lib.rs:895 cmd_postprocess_image(..., &self.denoising_images[0], ...). With DENOISE_PASSES = 8 the final a-trous pass writes to denoising_images[1] (pass_index % 2 == 1), so postprocess reads pass-6's result, not pass-7's. Existed in main too (was self.denoising_images[0].inner()), so not the black-screen cause — but worth fixing alongside (use
denoising_images[(DENOISE_PASSES - 1) % 2 ^ 1] or whatever the correct index is).