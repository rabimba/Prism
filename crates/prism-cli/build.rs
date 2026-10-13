use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    if let Some(workspace_root) = manifest_dir.parent().and_then(|p| p.parent()) {
        let llama_lib = workspace_root.join("Vendor").join("llama").join("lib");
        let pdfium_lib = workspace_root.join("Vendor").join("pdfium").join("lib");
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}:{}",
            llama_lib.display(),
            pdfium_lib.display()
        );
    }
}
