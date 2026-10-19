import os
import sys
import tempfile
import urllib.request
import urllib.error
import logging
from typing import Tuple, Optional

logger = logging.getLogger("webapp.model_loader")


def download_model_checkpoint(
    target_path: str,
    model_url: Optional[str] = None,
    timeout: float = 60.0,
    min_size_bytes: int = 1024,
) -> Tuple[bool, str]:
    """
    Safely download a PyTorch model checkpoint from a server-side configured URL.

    Security & Reliability Rules:
    - Never download from client-supplied URLs; only uses server-side environment
      variable MODEL_URL or trusted server configuration.
    - Uses a reasonable connection and read timeout (default 60s).
    - Downloads directly to a temporary file in the destination directory to ensure
      an atomic replace on the same filesystem.
    - Validates that the downloaded file is non-empty and meets min_size_bytes.
    - Cleans up temporary/partial files on any failure, ensuring no partial or
      corrupted file remains at target_path.
    """
    if os.path.exists(target_path):
        return True, f"Model checkpoint already exists at {target_path}."

    url = model_url or os.environ.get("MODEL_URL")
    if not url or not url.strip():
        return False, "MODEL_URL environment variable is not configured and local checkpoint is missing."

    url = url.strip()
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    # Use a temp file in the same directory to guarantee atomic replace across filesystems
    temp_fd, temp_path = tempfile.mkstemp(prefix="model_dl_", suffix=".tmp", dir=target_dir)
    os.close(temp_fd)

    logger.info("Downloading model checkpoint from %s to %s", url, target_path)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FlappyBird-DQN-Backend/1.0)"}
        )

        bytes_downloaded = 0
        chunk_size = 64 * 1024

        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            if status_code >= 400:
                raise ValueError(f"HTTP response error {status_code} from model URL.")

            with open(temp_path, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    bytes_downloaded += len(chunk)

        if bytes_downloaded < min_size_bytes:
            raise ValueError(
                f"Downloaded file is too small ({bytes_downloaded} bytes). Expected at least {min_size_bytes} bytes."
            )

        # Atomic replacement: replaces target_path if it exists or creates it atomically
        os.replace(temp_path, target_path)
        logger.info("Successfully downloaded and installed checkpoint (%d bytes)", bytes_downloaded)
        return True, f"Successfully downloaded model checkpoint ({bytes_downloaded} bytes)."

    except Exception as exc:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        err_msg = f"Failed to download model checkpoint from MODEL_URL: {exc}"
        logger.error(err_msg)
        return False, err_msg
