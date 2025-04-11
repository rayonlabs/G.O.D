import os
from urllib.parse import urlparse

import aiohttp


async def download_s3_file(file_url: str, save_path: str = None, tmp_dir: str = "/tmp") -> str:
    """Download a file from an S3 URL and save it locally.

    Args:
        file_url (str): The URL of the file to download.
        save_path (str, optional): The path where the file should be saved. If a directory is provided,
            the file will be saved with its original name in that directory. If a file path is provided,
            the file will be saved at that exact location. Defaults to None.
        tmp_dir (str, optional): The temporary directory to use when save_path is not provided.
            Defaults to "/tmp".

    Returns:
        str: The local file path where the file was saved.

    Raises:
        Exception: If the download fails with a non-200 status code.

    Example:
        >>> file_path = await download_s3_file("https://example.com/file.txt", save_path="/data")
        >>> print(file_path)
        /data/file.txt
    """
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    if save_path:
        if os.path.isdir(save_path):
            local_file_path = os.path.join(save_path, file_name)
        else:
            local_file_path = save_path
    else:
        local_file_path = os.path.join(tmp_dir, file_name)

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(local_file_path, "wb") as f:
                    f.write(await response.read())
            else:
                raise Exception(f"Failed to download file: {response.status}")

    return local_file_path
