use std::{ffi::CString, fs::File, io::Write};

use shader_slang as slang;
use shader_slang::Downcast;

/*
* This build script compiles shaders in the shaders/ directory into .spirv files under $OUT_DIR.
* GLSL shaders are compiled with `shaderc`. Slang shaders are compiled with `shader-slang`
* and emit SPIR-V with the `spvDescriptorHeapEXT` capability enabled (matches the runtime
* compiler in src/shader_compiler), so they plug straight into a VK_EXT_descriptor_heap pipeline.
 */

// this must be the same as what is specified in ray_tracing_pipeline.rs
const SHADER_ENTRY_POINT: &str = "main";

fn output_file_prefix(name: &str) -> String {
    format!("{}/{}", std::env::var("OUT_DIR").unwrap(), name)
}
fn input_file_prefix(name: &str) -> String {
    format!("{}/{}", std::env::var("CARGO_MANIFEST_DIR").unwrap(), name)
}

fn compile_shader(file_name: &str, shader_type: shaderc::ShaderKind, generate_debug_info: bool, out_file_name: &str) {
    let file_contents = std::fs::read_to_string(input_file_prefix(file_name))
        .unwrap_or_else(|e| panic!("while reading shader file '{file_name}': {e}"));

    //TODO: unwrap
    let compiler = shaderc::Compiler::new().unwrap();

    let mut options = shaderc::CompileOptions::new().unwrap();
    if generate_debug_info {
        options.set_generate_debug_info();
    }
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_4 as u32);
    options.set_include_callback(|included_file_name, _included_type, _including_file_name, _include_depth| {
        if _included_type == shaderc::IncludeType::Relative {
            panic!("Found relative include \"{included_file_name}\"; only standard include (#include <header>) is allowed");
        }

        let file_contents = std::fs::read_to_string(input_file_prefix(included_file_name))
            .unwrap_or_else(|e| panic!("while reading shader file '{included_file_name}', included from '{file_name}': {e}"));

        Ok(shaderc::ResolvedInclude {
            resolved_name: included_file_name.to_string(),
            content: file_contents.to_string(),
        })
    });

    let preprocessed = compiler
        .preprocess(&file_contents, file_name, SHADER_ENTRY_POINT, Some(&options))
        .unwrap_or_else(|e| panic!("Could not preprocess shader: {e}"))
        .as_text();

    let binary_result = compiler
        .compile_into_spirv(&preprocessed, shader_type, file_name, SHADER_ENTRY_POINT, Some(&options))
        .unwrap_or_else(|e| panic!("Could not preprocess shader: {e}"));

    let mut out_file = File::create(output_file_prefix(out_file_name))
        .unwrap_or_else(|e| panic!("While opening/creating shader spirv file '{out_file_name}' for write: {e}"));
    out_file
        .write_all(binary_result.as_binary_u8())
        .unwrap_or_else(|e| panic!("While writing to shader spirv file '{out_file_name}': {e}"));
}

/// Compile a Slang module to SPIR-V at build time. `module_name` is the file stem under
/// `shaders/` (no `.slang`). Mirrors the runtime compiler in `src/shader_compiler/compiler.rs`,
/// so the bytes the two paths produce are interchangeable.
fn compile_slang_shader(module_name: &str, entry_point: &str, out_file_name: &str) {
    let global_session = slang::GlobalSession::new()
        .expect("Failed to create Slang GlobalSession (is the Slang runtime DLL on PATH?)");

    let descriptor_heap_cap = global_session.find_capability("spvDescriptorHeapEXT");
    if descriptor_heap_cap.is_unknown() {
        panic!(
            "Slang does not know the `spvDescriptorHeapEXT` capability — \
             the installed Slang predates PR #10177 (Feb 2026). Update the Slang runtime."
        );
    }

    // Column-major matrix storage matches nalgebra's column-major Matrix4 on the
    // CPU (and GLSL's default), so reading `matrices.view_inverse * v` in the
    // Slang RT shaders produces the same result as the original GLSL.
    let session_options = slang::CompilerOptions::default()
        .optimization(slang::OptimizationLevel::High)
        .matrix_layout_row(false)
        .capability(descriptor_heap_cap);

    let target_desc = slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global_session.find_profile("spirv_1_6"));

    let targets = [target_desc];

    let shaders_dir = input_file_prefix("shaders");
    let search_path = CString::new(shaders_dir.clone())
        .unwrap_or_else(|e| panic!("shaders dir '{shaders_dir}' contains nul byte: {e}"));
    let search_paths = [search_path.as_ptr()];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = global_session
        .create_session(&session_desc)
        .expect("Slang create_session returned null");

    let module = session
        .load_module(module_name)
        .unwrap_or_else(|e| panic!("Slang load_module(\"{module_name}\") failed: {e}"));

    let entry = module
        .find_entry_point_by_name(entry_point)
        .unwrap_or_else(|| panic!("entry point \"{entry_point}\" not found in module \"{module_name}\""));

    let program = session
        .create_composite_component_type(&[module.downcast().clone(), entry.downcast().clone()])
        .unwrap_or_else(|e| panic!("Slang create_composite_component_type failed: {e}"));

    let linked = program
        .link()
        .unwrap_or_else(|e| panic!("Slang link failed for \"{module_name}::{entry_point}\": {e}"));

    let spirv_blob = linked
        .entry_point_code(0, 0)
        .unwrap_or_else(|e| panic!("Slang entry_point_code failed for \"{module_name}::{entry_point}\": {e}"));

    let mut out_file = File::create(output_file_prefix(out_file_name))
        .unwrap_or_else(|e| panic!("While opening/creating shader spirv file '{out_file_name}' for write: {e}"));
    out_file
        .write_all(spirv_blob.as_slice())
        .unwrap_or_else(|e| panic!("While writing to shader spirv file '{out_file_name}': {e}"));
}

fn main() {
    println!("cargo::rerun-if-changed=shaders/");

    compile_shader(
        "shaders/ray_gen_ris.glsl",
        shaderc::ShaderKind::RayGeneration,
        false,
        "ray_gen_ris.spirv",
    );
    compile_shader(
        "shaders/ray_gen_final.glsl",
        shaderc::ShaderKind::RayGeneration,
        false,
        "ray_gen_final.spirv",
    );
    compile_shader(
        "shaders/closest_hit.glsl",
        shaderc::ShaderKind::ClosestHit,
        false,
        "closest_hit.spirv",
    );
    compile_shader("shaders/any_hit.glsl", shaderc::ShaderKind::AnyHit, false, "any_hit.spirv");
    compile_shader("shaders/ray_miss.glsl", shaderc::ShaderKind::Miss, false, "ray_miss.spirv");
    compile_shader("shaders/denoise.glsl", shaderc::ShaderKind::Compute, false, "denoise.spirv");
    compile_shader(
        "shaders/temporal_accumulation.glsl",
        shaderc::ShaderKind::Compute,
        false,
        "temporal_accumulation.spirv",
    );
    compile_shader(
        "shaders/postprocess.glsl",
        shaderc::ShaderKind::Compute,
        false,
        "postprocess.spirv",
    );

    // Slang shaders. The runtime path in src/shader_compiler still compiles
    // postprocess.slang as well — we keep the build-time artifact so callers
    // can choose to skip the runtime hop.
    compile_slang_shader("postprocess", "main", "postprocess_slang.spirv");

    // Raytracing pipeline (heap mode). One Slang module per stage; the entry
    // point matches the [shader("…")] attribute inside each file.
    compile_slang_shader("ray_miss",    "ray_miss",    "ray_miss_slang.spirv");
    compile_slang_shader("any_hit",     "any_hit",     "any_hit_slang.spirv");
    compile_slang_shader("closest_hit", "closest_hit", "closest_hit_slang.spirv");
    compile_slang_shader("ray_gen_ris",   "ray_gen_ris",   "ray_gen_ris_slang.spirv");
    compile_slang_shader("ray_gen_final", "ray_gen_final", "ray_gen_final_slang.spirv");
}
