from huggingface_hub import list_repo_files
files = list_repo_files("keremberke/license-plate-object-detection")
print("\n".join(files))