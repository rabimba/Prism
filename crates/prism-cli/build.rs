use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    if let Some(workspace_root) = manifest_dir.parent().and_then(|p| p.parent()) {
        let llama_lib = workspace_root.join("Vendor").join("llama").join("lib");
        let pdfium_lib = workspace_root.join("Vendor").join("pdfium").join("lib");
        if cfg!(target_os = "macos") {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,@executable_path,-rpath,@executable_path/../Frameworks,-rpath,@executable_path/../lib/prism,-rpath,{},-rpath,{}",
                llama_lib.display(),
                pdfium_lib.display()
            );
        } else {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN:$ORIGIN/../lib/prism:$ORIGIN/../lib:{}:{}",
                llama_lib.display(),
                pdfium_lib.display()
            );
        }
    }
}
