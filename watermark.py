"""AES-GCM encryption and block-DCT watermarking helpers."""

from __future__ import annotations

import secrets
import struct
from pathlib import Path

import cv2
import numpy as np
from Crypto.Cipher import AES
from scipy.fft import dctn, idctn


ROOT_DIR = Path(__file__).resolve().parent
DIRECTORIES = ("input", "output", "input_check", "output_check")
KEY_PATH = ROOT_DIR / "secret.key"
MAGIC = b"DW01"
HEADER_SIZE = len(MAGIC) + 4
AES_NONCE_SIZE = 12
AES_TAG_SIZE = 16
BLOCK_SIZE = 8
REPETITIONS = 5
DCT_DELTA = 64.0
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_directories() -> None:
    for name in DIRECTORIES:
        (ROOT_DIR / name).mkdir(parents=True, exist_ok=True)


def load_or_create_key(key_path: Path = KEY_PATH) -> bytes:
    """Load the persistent AES-256 key, creating it on the first run."""
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if not key_path.exists():
        try:
            with key_path.open("xb") as key_file:
                key_file.write(secrets.token_bytes(32))
        except FileExistsError:
            pass

    key = key_path.read_bytes()
    if len(key) != 32:
        raise ValueError(f"{key_path.name} phải chứa đúng 32 byte (AES-256).")
    return key


def aes_encrypt(plaintext: str, key: bytes) -> bytes:
    """Encrypt UTF-8 text with AES-256-GCM; return version + nonce + tag + data."""
    if len(key) != 32:
        raise ValueError("Khóa AES-256 phải có đúng 32 byte.")
    cipher = AES.new(key, AES.MODE_GCM, nonce=secrets.token_bytes(AES_NONCE_SIZE))
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
    return MAGIC + cipher.nonce + tag + ciphertext


def aes_decrypt(encrypted: bytes, key: bytes) -> str:
    """Authenticate and decrypt an AES-GCM payload."""
    if len(key) != 32:
        raise ValueError("Khóa AES-256 phải có đúng 32 byte.")
    minimum_size = len(MAGIC) + AES_NONCE_SIZE + AES_TAG_SIZE
    if len(encrypted) < minimum_size or encrypted[: len(MAGIC)] != MAGIC:
        raise ValueError("Dữ liệu mã hóa không hợp lệ hoặc không đầy đủ.")

    nonce_start = len(MAGIC)
    tag_start = nonce_start + AES_NONCE_SIZE
    cipher = AES.new(
        key,
        AES.MODE_GCM,
        nonce=encrypted[nonce_start:tag_start],
    )
    try:
        plaintext = cipher.decrypt_and_verify(
            encrypted[tag_start + AES_TAG_SIZE :],
            encrypted[tag_start : tag_start + AES_TAG_SIZE],
        )
    except ValueError as exc:
        raise ValueError(
            "Xác thực AES-GCM thất bại: thủy vân bị hỏng hoặc khóa không đúng."
        ) from exc

    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Payload sau giải mã không phải UTF-8 hợp lệ.") from exc


def _bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def _bits_to_bytes(bits: list[int]) -> bytes:
    return bytes(
        sum(bits[index + offset] << (7 - offset) for offset in range(8))
        for index in range(0, len(bits), 8)
    )


def _get_luma(image: np.ndarray) -> tuple[np.ndarray, bool]:
    if image.ndim == 2:
        return image.astype(np.float32), False
    if image.ndim == 3 and image.shape[2] in (3, 4):
        ycrcb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 0].astype(np.float32), True
    raise ValueError("Ảnh phải là grayscale, BGR 3 kênh hoặc BGRA 4 kênh.")


def _write_luma(image: np.ndarray, luma: np.ndarray, is_color: bool) -> np.ndarray:
    height, width = luma.shape
    updated = np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    if not is_color:
        return updated

    ycrcb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2YCrCb)
    ycrcb[:height, :width, 0] = updated
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    if image.shape[2] == 4:
        return np.dstack((bgr, image[:, :, 3]))
    return bgr


def _block_positions(height: int, width: int):
    for top in range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE):
        for left in range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE):
            yield top, left


def _qim_target(difference: float, bit: int, delta: float) -> float:
    offset = 0.25 if bit == 0 else 0.75
    lattice_index = round(difference / delta - offset)
    return (lattice_index + offset) * delta


def _embed_bit(block: np.ndarray, bit: int, delta: float) -> np.ndarray:
    coefficients = dctn(block, type=2, norm="ortho")
    first, second = coefficients[2, 3], coefficients[3, 2]
    target = _qim_target(float(first - second), bit, delta)
    midpoint = (first + second) / 2.0
    coefficients[2, 3] = midpoint + target / 2.0
    coefficients[3, 2] = midpoint - target / 2.0
    return idctn(coefficients, type=2, norm="ortho")


def _read_bit(block: np.ndarray, delta: float) -> int:
    coefficients = dctn(block, type=2, norm="ortho")
    phase = float((coefficients[2, 3] - coefficients[3, 2]) % delta) / delta
    distance_zero = min(abs(phase - 0.25), 1.0 - abs(phase - 0.25))
    distance_one = min(abs(phase - 0.75), 1.0 - abs(phase - 0.75))
    return int(distance_one < distance_zero)


def embed_watermark_dct(
    image: np.ndarray,
    encrypted_payload: bytes,
    repetitions: int = REPETITIONS,
    delta: float = DCT_DELTA,
) -> np.ndarray:
    """Embed encrypted bytes in repeated mid-frequency DCT coefficient pairs."""
    if repetitions < 1 or delta <= 0:
        raise ValueError("repetitions và delta phải là số dương.")
    if len(encrypted_payload) > 0xFFFFFFFF:
        raise ValueError("Payload quá lớn.")

    frame = MAGIC + struct.pack(">I", len(encrypted_payload)) + encrypted_payload
    bits = _bytes_to_bits(frame)
    luma, is_color = _get_luma(image)
    positions = list(_block_positions(*luma.shape))
    required_blocks = len(bits) * repetitions
    if len(positions) < required_blocks:
        capacity_bytes = max(0, len(positions) // repetitions // 8 - HEADER_SIZE)
        raise ValueError(
            "Ảnh không đủ dung lượng DCT cho payload này "
            f"(ước tính tối đa {capacity_bytes} byte payload mã hóa). "
            "Hãy dùng ảnh lớn hơn hoặc rút ngắn nội dung bản quyền."
        )

    for block_index, (top, left) in enumerate(positions):
        bit_index = (block_index // repetitions) % len(bits)
        luma[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE] = _embed_bit(
            luma[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE],
            bits[bit_index],
            delta,
        )
    return _write_luma(image, luma, is_color)


def extract_watermark_dct(
    image: np.ndarray,
    repetitions: int = REPETITIONS,
    delta: float = DCT_DELTA,
) -> bytes:
    """Extract and majority-vote an encrypted payload from a DCT-watermarked image."""
    if repetitions < 1 or delta <= 0:
        raise ValueError("repetitions và delta phải là số dương.")
    luma, _ = _get_luma(image)
    raw_bits = [
        _read_bit(luma[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE], delta)
        for top, left in _block_positions(*luma.shape)
    ]

    header_bits_count = HEADER_SIZE * 8
    if len(raw_bits) < header_bits_count * repetitions:
        raise ValueError("Ảnh không còn đủ dữ liệu để đọc phần đầu thủy vân.")

    header_bits = [
        int(
            sum(
                raw_bits[bit_index * repetitions : (bit_index + 1) * repetitions]
            )
            > repetitions / 2
        )
        for bit_index in range(header_bits_count)
    ]
    header = _bits_to_bytes(header_bits)
    if header[: len(MAGIC)] != MAGIC:
        raise ValueError("Không tìm thấy thủy vân DCT hợp lệ trong ảnh.")

    encrypted_size = struct.unpack(">I", header[len(MAGIC) : HEADER_SIZE])[0]
    total_bits = (HEADER_SIZE + encrypted_size) * 8
    if len(raw_bits) < total_bits * repetitions:
        raise ValueError(
            "Ảnh bị cắt hoặc hư hỏng: không đủ dữ liệu để khôi phục payload."
        )

    votes: list[list[int]] = [[] for _ in range(total_bits)]
    for block_index, bit in enumerate(raw_bits):
        bit_index = (block_index // repetitions) % total_bits
        votes[bit_index].append(bit)

    recovered_bits = [
        int(sum(bit_votes) > len(bit_votes) / 2)
        for bit_votes in votes
    ]
    recovered_frame = _bits_to_bytes(recovered_bits)
    if recovered_frame[: len(MAGIC)] != MAGIC:
        raise ValueError("Phần đầu thủy vân không nhất quán sau khi bỏ phiếu.")
    return recovered_frame[HEADER_SIZE : HEADER_SIZE + encrypted_size]


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
