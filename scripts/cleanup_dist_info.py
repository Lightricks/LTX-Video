"""
Clean up duplicate dist-info directories that can cause version conflicts.

This is needed because Pinokio's drive linking system can create multiple
dist-info symlinks for the same package with different versions.
"""
import os
import pathlib
import importlib.metadata

def get_site_packages():
    """Find the site-packages directory in the virtual environment."""
    env_path = pathlib.Path(__file__).parent.parent / "env" / "lib"
    for p in env_path.glob("python*/site-packages"):
        return p
    return None

def cleanup_duplicate_dist_info():
    site_packages = get_site_packages()
    if not site_packages:
        print("Could not find site-packages directory")
        return

    # Packages known to have version conflicts
    packages_to_check = ["huggingface_hub", "transformers", "tokenizers"]

    for pkg_name in packages_to_check:
        try:
            installed_version = importlib.metadata.version(pkg_name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            continue

        # Find all dist-info directories for this package
        pattern = f"{pkg_name}-*.dist-info"
        alt_pattern = f"{pkg_name.replace('_', '-')}-*.dist-info"

        dist_infos = list(site_packages.glob(pattern)) + list(site_packages.glob(alt_pattern))

        for dist_info in dist_infos:
            # Extract version from dist-info name
            name_parts = dist_info.name.replace(".dist-info", "").split("-")
            if len(name_parts) >= 2:
                dist_version = name_parts[1]

                # Remove if version doesn't match and it's a symlink
                if dist_version != installed_version and dist_info.is_symlink():
                    print(f"Removing duplicate: {dist_info.name} (installed: {installed_version})")
                    dist_info.unlink()

if __name__ == "__main__":
    cleanup_duplicate_dist_info()
    print("Cleanup complete")
