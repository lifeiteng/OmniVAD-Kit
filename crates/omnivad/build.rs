use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=OMNIVAD_LIB_DIR");
    println!("cargo:rerun-if-changed=../../native/build");

    if let Some(dir) = find_lib_dir() {
        let dir = dir.canonicalize().unwrap_or(dir);
        if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
}

fn find_lib_dir() -> Option<PathBuf> {
    if let Ok(dir) = env::var("OMNIVAD_LIB_DIR") {
        return Some(resolve_lib_dir(PathBuf::from(dir)));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .ancestors()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.clone());

    [
        workspace_root.join("native/build"),
        workspace_root.join("omnivad"),
        workspace_root.join("build"),
    ]
    .into_iter()
    .find(|dir| contains_omnivad_library(dir.as_path()))
}

fn contains_omnivad_library(dir: &Path) -> bool {
    [
        "libomnivad.dylib",
        "libomnivad.so",
        "libomnivad.a",
        "omnivad.dll",
        "omnivad.lib",
    ]
    .iter()
    .any(|name| dir.join(name).exists())
}

fn resolve_lib_dir(dir: PathBuf) -> PathBuf {
    if dir.is_absolute() || dir.exists() {
        return dir;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_relative = manifest_dir.ancestors().nth(2).map(|root| root.join(&dir));
    if let Some(path) = workspace_relative {
        if path.exists() {
            return path;
        }
    }

    dir
}
