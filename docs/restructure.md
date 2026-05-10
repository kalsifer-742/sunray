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

4. Pipeline layout changes — every pipeline gets the heap-creation flag and a push-constant range; shaders rewritten to read from the heap.
5. Migrate passes one at a time, delete descriptor_sets/ last.Fix