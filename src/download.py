from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="openai/clip-vit-large-patch14", filename="config.json", cache_dir="../clip/clip-vit-large-patch14")