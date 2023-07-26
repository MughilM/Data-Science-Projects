"""
File: utils/general_funcs.py
Creation Date: 2023-07-14

Just general use functions that can be used anywhere, maybe to simplify an annoying workflow.
"""
import os
import requests
from tqdm.auto import tqdm

def download_file_url(url, directory, filename):
    """
    Used to download large files given a direct URL.
    Uses requests package.
    :param url:
    :param directory:
    :param filename:
    :return:
    """
    # General code from tqdm's example at https://github.com/tqdm/tqdm/blob/master/examples/tqdm_requests.py.
    full_path = os.path.join(directory, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(full_path),
                      total=int(r.headers.get('content-length', 0))) as progress_bar:
                for chunk in r.iter_content(chunk_size=4096):
                    f.write(chunk)
                    progress_bar.update(len(chunk))  # The last chunk might not be 4096.
    return full_path
