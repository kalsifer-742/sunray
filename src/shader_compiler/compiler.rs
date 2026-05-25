use std::ffi::CString;
use std::path::PathBuf;

use shader_slang as slang;
use shader_slang::Downcast;

use crate::error::{SrError, SrResult};

/// Slang compiler bound to a single shaders directory. Cheap to keep alive — the
/// expensive object is the `GlobalSession`, which we hold for the renderer's lifetime.
/// Each `compile()` call spins up a fresh `Session` so options stay independent
/// per-compile (good enough for the first iteration; per-stage caching can come later).
pub struct ShaderCompiler {
    global_session: slang::GlobalSession,
    descriptor_heap_cap: slang::CapabilityID,
    /// CString-stored to keep the `*const i8` we hand to `SessionDesc::search_paths` valid.
    search_path: CString,
}

impl ShaderCompiler {
    pub fn new(shaders_dir: PathBuf) -> SrResult<Self> {
        let global_session = slang::GlobalSession::new().ok_or_else(|| {
            SrError::new_custom("Failed to create Slang GlobalSession (is the Slang runtime DLL on PATH?)".into())
        })?;

        let descriptor_heap_cap = global_session.find_capability("spvDescriptorHeapEXT");
        if descriptor_heap_cap.is_unknown() {
            return Err(SrError::new_custom(
                "Slang does not know the `spvDescriptorHeapEXT` capability — \
                 the installed Slang predates PR #10177 (Feb 2026). Update the Slang runtime."
                    .into(),
            ));
        }

        let dir_str = shaders_dir
            .to_str()
            .ok_or_else(|| SrError::new_custom(format!("non-utf8 shaders dir: {shaders_dir:?}")))?;
        let search_path =
            CString::new(dir_str).map_err(|e| SrError::new_custom(format!("shaders dir contains nul byte: {e}")))?;

        Ok(Self {
            global_session,
            descriptor_heap_cap,
            search_path,
        })
    }

    /// Compiles a Slang module + entry point to SPIR-V bytes ready for `vk::ShaderModuleCreateInfo`.
    /// `module_name` is the file stem under the shaders dir (no `.slang`); the entry point is
    /// looked up by name on that module.
    pub fn compile(&self, module_name: &str, entry_point: &str) -> SrResult<Vec<u8>> {
        let session_options = slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::High)
            .matrix_layout_row(true)
            .capability(self.descriptor_heap_cap);

        let target_desc = slang::TargetDesc::default()
            .format(slang::CompileTarget::Spirv)
            .profile(self.global_session.find_profile("spirv_1_5"));

        let targets = [target_desc];
        let search_paths = [self.search_path.as_ptr()];

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_paths)
            .options(&session_options);

        let session = self
            .global_session
            .create_session(&session_desc)
            .ok_or_else(|| SrError::new_custom("Slang create_session returned null".into()))?;

        let module = session
            .load_module(module_name)
            .map_err(|e| SrError::new_custom(format!("Slang load_module(\"{module_name}\") failed: {e}")))?;

        let entry = module
            .find_entry_point_by_name(entry_point)
            .ok_or_else(|| SrError::new_custom(format!("entry point \"{entry_point}\" not found in module \"{module_name}\"")))?;

        let program = session
            .create_composite_component_type(&[module.downcast().clone(), entry.downcast().clone()])
            .map_err(|e| SrError::new_custom(format!("Slang create_composite_component_type failed: {e}")))?;

        let linked = program
            .link()
            .map_err(|e| SrError::new_custom(format!("Slang link failed: {e}")))?;

        let spirv_blob = linked.entry_point_code(0, 0).map_err(|e| {
            SrError::new_custom(format!(
                "Slang entry_point_code failed for \"{module_name}::{entry_point}\": {e}"
            ))
        })?;

        Ok(spirv_blob.as_slice().to_vec())
    }
}
