use std::{fs::File, io::Write};

/*
* This build script has the sole purpose of compiling glsl shaders in the shaders/ directory, turning them into .spirv files under $OUT_DIR
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

fn main() {
    println!("cargo::rerun-if-changed=shaders/");

    compile_shader(
        "shaders/ray_gen.glsl",
        shaderc::ShaderKind::RayGeneration,
        false,
        "ray_gen.spirv",
    );
    compile_shader(
        "shaders/closest_hit.glsl",
        shaderc::ShaderKind::ClosestHit,
        false,
        "closest_hit.spirv",
    );
    compile_shader("shaders/ray_miss.glsl", shaderc::ShaderKind::Miss, false, "ray_miss.spirv");
}
