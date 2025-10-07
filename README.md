# sunray

Rust hardware ray-tracing library

![prova](/docs//render.png)

This project was developed by [kalsifer-742](https://github.com/kalsifer-742) and [@circled-square](https://github.com/circled-square) supervised by Professor [Marco Patrignani](https://squera.github.io/) for the bachelor thesis at the University of Trento, Italy

## Contribution

## Comparison

|                                                                      | Active project | Non-trivial | Real-time | Fully ray-traced | Hybrid |  GPU  | HW RT | Compute | SIMD  |  BVH  | Mesh  | Materials | Denoise | Rust  | Crate |  Engine   |                         Notes |
| :------------------------------------------------------------------- | :------------: | :---------: | :-------: | :--------------: | :----: | :---: | :---: | :-----: | :---: | :---: | :---: | :-------: | :-----: | :---: | :---: | :-------: | ----------------------------: |
| [Kajiya](https://github.com/EmbarkStudios/kajiya)                    |       ❌        |      ✅      |     ✅     |        ✅         |   ✅    |   ✅   |   ✅   |    ✅    |   ❌   |   ?   |   ✅   |     ✅     |    ✅    |   ✅   |   ❌   |     ❌     |                               |
| [Cycles](https://projects.blender.org/blender/cycles)                |       ✅        |      ✅      |     ❌     |        ✅         |   ❌    |   ✅   |   ✅   |    ✅    |   ✅   |   ✅   |   ✅   |     ✅     |    ✅    |   ❌   |  N/A  | ✅ Blender |                               |
| [manta-ray](https://github.com/ange-yaghi/manta-ray)                 |       ❌        |      ✅      |     ❌     |        ✅         |   ❌    |   ✅   |   ❌   |    ✅    |   ✅   |   ✅   |   ✅   |     ✅     |    ✅    |   ❌   |  N/A  | ✅ Blender |                               |
| [luxcore](https://luxcorerender.org/)                                |       ✅        |      ✅      |     ❌     |        ?         |   ?    |   ✅   |   ❌   |    ✅    |   ?   |   ?   |   ✅   |     ✅     |    ?    |   ❌   |  N/A  | ✅ Blender |                               |
| [akari_render](https://github.com/shiinamiyuki/akari_render)         |       ❌        |      ✅      |     ?     |        ?         |   ?    |   ✅   |   ❌   |    ✅    |   ?   |   ?   |   ✅   |     ✅     |    ?    |   ✅   |   ❌   | ✅ Blender |    Rebuild blender to install |
| [rustracer](https://github.com/KaminariOS/rustracer)                 |       ❌        |      ✅      |     ❌     |        ✅         |   ❌    |   ✅   |   ✅   |    ❌    |   ❌   |   ❌   |   ✅   |     ✅     |    ❌    |   ✅   |   ❌   |     ❌     |                      uses Nix |
| [RayTracingInVulkan](https://github.com/GPSnoopy/RayTracingInVulkan) |       ✅        |      ✅      |     ✅     |        ✅         |   ❌    |   ✅   |   ✅   |    ❌    |   ?   |   ✅   |   ✅   |  partial  |    ❌    |   ❌   |  N/A  |     ❌     |                               |
| [referencePT](https://github.com/boksajak/referencePT)               |       ❌        |      ✅      |     ?     |        ?         |   ?    |   ✅   |   ✅   |    ❌    |   ❌   |   ?   |   ✅   |     ✅     |    ?    |   ❌   |  N/A  |     ❌     |                               |
| [gbrt](https://github.com/giulianbiolo/gbrt)                         |       ❌        |      ❌      |     ❌     |        ❌         |   ❌    |   ❌   |   ❌   |    ❌    |   ✅   |   ✅   |   ✅   |     ❌     |    ❌    |   ✅   |   ❌   |     ❌     |                               |
| [Godot4-RayTracing](https://github.com/bitegw/Godot4-Raytracing)     |       ❌        |      ❌      |     ✅     |        ✅         |   ❌    |   ✅   |   ❌   |    ✅    |   ❌   |   ❌   |   ❌   |  partial  |    ❌    |   ❌   |  N/A  |  ✅ Godot  |                               |
| [Raytracing_Godot4](https://github.com/nekotogd/Raytracing_Godot4)   |       ❌        |      ❌      |     ✅     |        ✅         |   ❌    |   ✅   |   ❌   |    ✅    |   ❌   |   ❌   |   ❌   |     ❌     |    ❌    |   ❌   |  N/A  |  ✅ Godot  |                               |
| [bevyray](https://github.com/GrandmasterB42/bevyray)                 |       ✅        |      ❌      |     ✅     |        ❌         |   ✅    |   ✅   |   ❌   |    ❌    |   ❌   |   ✅   |   ❌   |  partial  |    ❌    |   ✅   |   ❌   |  ✅ Bevy   | raytracing in fragment shader |
| [hanamaru-renderer](https://github.com/gam0022/hanamaru-renderer)    |       ❌        |      ❌      |     ❌     |        ?         |   ?    |   ❌   |   ❌   |    ❌    |   ?   |   ✅   |   ✅   |     ✅     |    ✅    |   ✅   |   ❌   |     ❌     |          docs are in japanese |
| [rtwlib](https://crates.io/crates/rtwlib)                            |       ✅        |      ❌      |     ❌     |        ✅         |   ❌    |   ❌   |   ❌   |    ❌    |   ❌   |   ❌   |   ❌   |     ❌     |    ❌    |   ✅   |   ✅   |     ❌     |                               |
| [rustic-zen](https://crates.io/crates/rustic-zen)                    |       ❌        |      ❌      |     ✅     |        ?         |   ?    |   ❌   |   ❌   |    ❌    |   ?   |   ?   |   ?   |     ?     |    ?    |   ✅   |   ✅   |     ❌     |                            2D |
| [rustracer](https://crates.io/crates/rustracer)                      |       ❌        |      ❌      |     ❌     |        ✅         |   ❌    |   ❌   |   ❌   |    ❌    |   ❌   |   ❌   |   ❌   |     ❌     |    ❌    |   ✅   |   ✅   |     ❌     |                               |
|                                                                      |                |             |           |                  |        |       |       |         |       |       |       |           |         |       |       |           |                               |
| [sunray](https://github.com/Kalsifer-742/sunray)                     |       ✅        |      ✅      |     ✅     |        ✅         |   ❌    |   ✅   |   ✅   |    ❌    |   ❌   |   ✅   |   ✅   |  partial  |    ❌    |   ✅   |   ✅   |     ❌     |                               |

## Resources

### General

- [Nvidia tutorial on vulkan KHR raytracing](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)
- [SaschaWillems basic ray tracing tutorial (C++)](https://github.com/SaschaWillems/Vulkan/blob/master/examples/raytracingbasic/raytracingbasic.cpp)
- [SaschaWillems vulkan tutorials (C++)](https://github.com/SaschaWillems/Vulkan)
- [Khronos vulkan samples (C++)](https://github.com/KhronosGroup/Vulkan-Samples/tree/main)
- [Ray Tracing in One Weekend - series](https://raytracing.github.io/)
- #### Other projects
  - [hatoo/ash-raytracing-example (Rust)](https://github.com/hatoo/ash-raytracing-example)
  - [adrien-ben/vulkan-examples-rs (Rust)](https://github.com/adrien-ben/vulkan-examples-rs)

### Rendering
- [Ray Tracing Gems II](https://developer.nvidia.com/ray-tracing-gems-ii)
- [pbrt](https://pbrt.org/)
  - [book](https://pbr-book.org/)
- [PBR for materials](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.pdf)
  - page 197 - appendix B: BRDF Implementation
- #### Shaders
  - https://www.gsn-lib.org/docs/nodes/raytracing.php
  - ##### Languages
    - [shader languages comparisons](https://alain.xyz/blog/a-review-of-shader-languages)
    - [slang](https://shader-slang.org/)
  - ##### Compilation
    - https://github.com/google/shaderc-rs

### Acceleration structure
- see [this nvidia blog](https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/) for best practices for acceleration structures (and hit shading)

### glTF
- [2.0 reference guide pdf](https://www.khronos.org/files/gltf20-reference-guide.pdf)
- [2.0 spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.pdf)
- [khronos tutorials](https://github.com/KhronosGroup/glTF-Tutorials/tree/main)
- https://www.gltfeditor.com/
- #### Extensions
  - [KHR_lights_punctual](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md)
- #### Models
  - [Khronos sample assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main)
  - [Lantern](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Lantern/README.md)

### Performance
- https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/
  - #### Syncronization
    - https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
    - https://xanderbert.github.io/2025/04/13/VulkanMemoryBarriers.html
    - https://cpp-rendering.io/barriers-vulkan-not-difficult/
    - [gpuopne - vulkan barriers explained](https://gpuopen.com/learn/vulkan-barriers-explained/)
    - [khr blog on image layout](https://www.khronos.org/blog/so-long-image-layouts-simplifying-vulkan-synchronisation)
  - #### Memory allocation
    - https://blog.io7m.com/2023/11/11/vulkan-memory-allocation.xhtml
    - https://github.com/Traverse-Research/gpu-allocator
    - https://github.com/gwihlidal/vk-mem-rs
    - https://docs.vulkan.org/guide/latest/memory_allocation.html
  - #### Queues
    - https://gpuopen.com/learn/concurrent-execution-asynchronous-queues/

### Miscelleaneus
- [Semantic Versioning 2.0.0](https://semver.org/)
- #### Coordinate Systems
  - [nalgebra computer-graphics recipes](https://nalgebra.rs/docs/user_guide/cg_recipes)
  - https://learnopengl.com/Getting-started/Coordinate-Systems
- #### Rust
  - https://doc.rust-lang.org/book/
  - https://doc.rust-lang.org/rust-by-example/
- #### Vulkan
  - [docs](https://docs.vulkan.org/guide/latest/index.html)
  - [tutorial](https://docs.vulkan.org/tutorial/latest/00_Introduction.html)
  - [unofficaila tutorial](https://vulkan-tutorial.com/)
  - [paminerva tutorial](https://paminerva.github.io/docs/LearnVulkan/LearnVulkan)