use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn copy_shared_libs(src_dir: &Path, dst_dir: &Path) {
    let Ok(entries) = fs::read_dir(src_dir) else {
        return;
    };
    let _ = fs::create_dir_all(dst_dir);
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.contains(".so") || name.ends_with(".dylib") {
            let dest = dst_dir.join(name);
            if !dest.exists() {
                let _ = fs::copy(&path, &dest);
            }
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("Failed to locate workspace root")
        .to_path_buf();

    let llama_dir = workspace_root.join("Vendor").join("llama");
    let llama_include = llama_dir.join("include");
    let llama_lib = llama_dir.join("lib");
    let pdfium_lib = workspace_root.join("Vendor").join("pdfium").join("lib");

    println!("cargo:rerun-if-changed=src/llama_shim.c");
    println!("cargo:rerun-if-changed={}", llama_include.display());

    let llama_lib_quoted = format!("\"{}\"", llama_lib.display());
    cc::Build::new()
        .file("src/llama_shim.c")
        .include(&llama_include)
        .define("PRISM_LLAMA_LIB_DIR", llama_lib_quoted.as_str())
        .opt_level(3)
        .warnings(false)
        .compile("prism_llama_shim");

    println!("cargo:rustc-link-search=native={}", llama_lib.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");
    println!("cargo:rustc-link-lib=dylib=ggml");

    // Stage shared libraries into target/<profile> and target/<profile>/deps so all
    // workspace binaries and test executables resolve via $ORIGIN / @executable_path.
    if let Ok(out_dir) = env::var("OUT_DIR") {
        let out_path = PathBuf::from(out_dir);
        if let Some(profile_dir) = out_path.ancestors().nth(3) {
            copy_shared_libs(&llama_lib, profile_dir);
            copy_shared_libs(&llama_lib, &profile_dir.join("deps"));
            if pdfium_lib.exists() {
                copy_shared_libs(&pdfium_lib, profile_dir);
                copy_shared_libs(&pdfium_lib, &profile_dir.join("deps"));
            }
        }
    }
}
