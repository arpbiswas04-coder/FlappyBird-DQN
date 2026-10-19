import os
import sys
import tempfile
import zipfile
import urllib.request
import urllib.error
import logging
from typing import Tuple, Optional

logger = logging.getLogger("webapp.model_loader")


def _is_zip_archive(file_path: str, url: str) -> bool:
    """Check if the downloaded file is a ZIP archive by extension or magic bytes."""
    if url.lower().split("?")[0].endswith(".zip"):
        return True
    try:
        return zipfile.is_zipfile(file_path)
    except Exception:
        return False


def _safe_extract_checkpoint_from_zip(
    zip_path: str,
    target_path: str,
    min_size_bytes: int = 1024,
) -> Tuple[bool, str]:
    """
    Safely extract the expected model checkpoint from a ZIP archive.

    Security & Safety:
    - Never uses extractall() or passes arbitrary archive member paths to disk.
    - Inspects archive members and identifies the file matching the target filename.
    - Extracts data directly into a temporary file in target_dir to avoid Zip Slip.
    - Validates that the extracted checkpoint is non-empty and meets min_size_bytes.
    - Atomically replaces target_path only after verification.
    """
    target_filename = os.path.basename(target_path)
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    temp_extract_path = None
    try:
        if not zipfile.is_zipfile(zip_path):
            return False, "Downloaded file is not a valid ZIP archive."

        with zipfile.ZipFile(zip_path, "r") as zf:
            matching_member = None
            for member_info in zf.infolist():
                if member_info.is_dir():
                    continue
                member_basename = os.path.basename(member_info.filename.replace("\\", "/"))
                if member_basename == target_filename:
                    matching_member = member_info
                    break

            if not matching_member:
                available_files = [
                    os.path.basename(m.filename.replace("\\", "/"))
                    for m in zf.infolist()
                    if not m.is_dir()
                ]
                return False, (
                    f"Target checkpoint '{target_filename}' not found in ZIP archive. "
                    f"Found files: {available_files}"
                )

            if matching_member.file_size < min_size_bytes:
                return False, (
                    f"Checkpoint file '{target_filename}' inside archive is too small "
                    f"({matching_member.file_size} bytes). Expected at least {min_size_bytes} bytes."
                )

            # Create temporary extraction file in target directory for atomic replace
            temp_extract_fd, temp_extract_path = tempfile.mkstemp(
                prefix="model_extract_", suffix=".tmp", dir=target_dir
            )
            os.close(temp_extract_fd)

            extracted_bytes = 0
            chunk_size = 64 * 1024
            with zf.open(matching_member, "r") as src, open(temp_extract_path, "wb") as dst:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    extracted_bytes += len(chunk)

            if extracted_bytes < min_size_bytes:
                return False, (
                    f"Extracted checkpoint '{target_filename}' is too small "
                    f"({extracted_bytes} bytes). Expected at least {min_size_bytes} bytes."
                )

            # Atomic replace
            os.replace(temp_extract_path, target_path)
            temp_extract_path = None  # Replaced successfully

            logger.info("Successfully extracted '%s' (%d bytes)", target_filename, extracted_bytes)
            return True, f"Successfully extracted '{target_filename}' ({extracted_bytes} bytes)."

    except zipfile.BadZipFile as bzf:
        return False, f"Corrupted or invalid ZIP archive: {bzf}"
    except Exception as exc:
        return False, f"Failed to extract checkpoint from ZIP: {exc}"
    finally:
        if temp_extract_path and os.path.exists(temp_extract_path):
            try:
                os.remove(temp_extract_path)
            except OSError:
                pass


def download_model_checkpoint(
    target_path: str,
    model_url: Optional[str] = None,
    timeout: float = 60.0,
    min_size_bytes: int = 1024,
) -> Tuple[bool, str]:
    """
    Safely download a PyTorch model checkpoint (raw .pt or .zip archive)
    from a server-side configured URL.

    Security & Reliability Rules:
    - Never download from client-supplied URLs; only uses server-side environment
      variable MODEL_URL or trusted server configuration.
    - Uses a reasonable connection and read timeout (default 60s).
    - Preserves existing valid local checkpoints without downloading anything.
    - Downloads directly to a temporary file in the destination directory.
    - If the downloaded file is a ZIP archive, safely extracts target_path
      preventing any path traversal attacks.
    - Validates that the downloaded/extracted file is non-empty and meets min_size_bytes.
    - Cleans up temporary/partial files on any failure, ensuring no partial or
      corrupted file remains at target_path.
    """
    # 1. Check if valid local checkpoint already exists
    if os.path.exists(target_path):
        existing_size = os.path.getsize(target_path)
        if existing_size >= min_size_bytes:
            return True, f"Model checkpoint already exists at {target_path} ({existing_size} bytes)."
        logger.warning(
            "Existing checkpoint at %s is smaller than minimum threshold (%d < %d bytes). Re-downloading.",
            target_path,
            existing_size,
            min_size_bytes,
        )

    # 2. Validate URL configuration
    url = model_url or os.environ.get("MODEL_URL")
    if not url or not url.strip():
        return False, "MODEL_URL environment variable is not configured and local checkpoint is missing."

    url = url.strip()
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    # Use a temp file in the same directory to guarantee atomic replace across filesystems
    temp_dl_fd, temp_dl_path = tempfile.mkstemp(prefix="model_dl_", suffix=".tmp", dir=target_dir)
    os.close(temp_dl_fd)

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

            with open(temp_dl_path, "wb") as out_file:
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

        # 3. Check if payload is a ZIP archive or direct model checkpoint
        if _is_zip_archive(temp_dl_path, url):
            extract_success, extract_msg = _safe_extract_checkpoint_from_zip(
                zip_path=temp_dl_path,
                target_path=target_path,
                min_size_bytes=min_size_bytes,
            )
            # Remove temporary zip file
            if os.path.exists(temp_dl_path):
                os.remove(temp_dl_path)

            if not extract_success:
                return False, f"ZIP extraction failed: {extract_msg}"

            return True, f"Successfully downloaded and extracted checkpoint from ZIP ({extract_msg})."

        # 4. Direct PyTorch checkpoint (.pt file)
        os.replace(temp_dl_path, target_path)
        logger.info("Successfully downloaded and installed checkpoint (%d bytes)", bytes_downloaded)
        return True, f"Successfully downloaded model checkpoint ({bytes_downloaded} bytes)."

    except Exception as exc:
        if os.path.exists(temp_dl_path):
            try:
                os.remove(temp_dl_path)
            except OSError:
                pass
        err_msg = f"Failed to download model checkpoint from MODEL_URL: {exc}"
        logger.error(err_msg)
        return False, err_msg
