{
  pkgs ? import <nixpkgs> { },
}:

let
  x11Libs = with pkgs; [
    libX11
    libXcursor
    libXrandr
    libXi
  ];
  waylandLibs = with pkgs; [
    wayland
    libxkbcommon
  ];
  vulkanLibs = with pkgs; [
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers
    vulkan-caps-viewer
  ];

  bpy-libs = with pkgs; [
    stdenv.cc.cc.lib # Standard C++ library
    zlib
    libGL
    libSM
    libICE
    libX11
    libXi
    libXxf86vm
    libXfixes
    libXrender
    wayland
    libxkbcommon
  ];
in
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    pkg-config
  ];

  buildInputs =
    with pkgs;
    [
      cmake
      clang
      llvmPackages.libclang
      glslang
      shaderc
      shader-slang
    ]
    ++ x11Libs
    ++ waylandLibs
    ++ vulkanLibs;

  # 1. Force shaderc-sys to use the pre-compiled Nix library (from previous step)
  SHADERC_LIB_DIR = "${pkgs.shaderc.lib}/lib";

  # 2. Tell the linker and runtime exactly where to find Vulkan and Windowing libraries
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
    bpy-libs ++ x11Libs ++ waylandLibs ++ vulkanLibs ++ [ pkgs.shader-slang ]
  );

  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  # 3. Use the nixpkgs Slang compiler instead of the one bundled with the Vulkan SDK.
  #    Headers live in the `dev` output, the shared libs in the default output.
  SLANG_INCLUDE_DIR = "${pkgs.shader-slang.dev}/include";
  SLANG_LIB_DIR = "${pkgs.shader-slang}/lib";

  VULKAN_SDK = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";

  VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
}
