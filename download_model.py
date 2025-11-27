"""
download_model.py

Download a GGUF model from Hugging Face with authentication.
Edit the top variables before running.
"""

import os
import requests
from tqdm import tqdm

# ---------------- USER CONFIG ----------------
HF_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HF_MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_0.gguf"
MODEL_DOWNLOAD_PATH = r"C:/Users/LAP14/Downloads/mistral-7b-openorca.Q4_0.gguf"
# ---------------------------------------------

def download_with_auth(url: str, output_path: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code != 200:
        print("DOWNLOAD FAILED:", response.status_code, response.text[:500])
        return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024
    print(f"Downloading to: {output_path} ({round(total_size/(1024*1024),2)} MB)")
    with open(output_path, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print("Download completed:", output_path)
    return True

if __name__ == "__main__":
    print("Starting download...")
    download_with_auth(HF_MODEL_URL, MODEL_DOWNLOAD_PATH, HF_API_TOKEN)
