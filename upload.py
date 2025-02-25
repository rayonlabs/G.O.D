import asyncio
import os
from validator.utils.minio import async_minio_client
from collections import defaultdict

async def upload_file(bucket: str, file_path: str):
    """
    Uploads a file to MinIO and returns its presigned URL.
    """
    upload_name = os.path.basename(file_path)
    await async_minio_client.upload_file(bucket, upload_name, file_path)
    return await async_minio_client.get_presigned_url(bucket, upload_name)

async def upload_folder(folder_path: str, bucket: str):
    """
    Loops through a folder, uploads paired text and image files, and returns structured results.
    """
    file_groups = defaultdict(dict)
    results = []

    # Group files by name (without extension)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            base_name, ext = os.path.splitext(file_name)
            if ext.lower() in {".txt"}:
                file_groups[base_name]["text"] = file_path
            elif ext.lower() in {".jpg", ".jpeg", ".png", ".webp"}:  # Add other image formats if needed
                file_groups[base_name]["image"] = file_path
    
    # Upload files and structure results
    for base_name, files in file_groups.items():
        text_url = await upload_file(bucket, files["text"]) if "text" in files else None
        image_url = await upload_file(bucket, files["image"]) if "image" in files else None
        
        if text_url and image_url:
            results.append({"image_url": image_url, "text_url": text_url})
    
    return results

# def main(folder_path: str, bucket: str = "GODTesting"):
def main():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # results = loop.run_until_complete(upload_folder(folder_path, bucket))
    results = loop.run_until_complete(upload_file("GODTesting", "/root/G.O.D/validator/evaluation/ComfyUI/output/hamza.zip"))
    print(results)
    return results


if __name__ == "__main__":
    # folder_path = "/root/glitch_art_lora"  # Replace with the actual folder path
    # main(folder_path)
    main()
