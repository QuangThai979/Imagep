"""Embed encrypted copyright text into an image from input/."""

from pathlib import Path

import cv2

from watermark import (
    ROOT_DIR,
    aes_encrypt,
    embed_watermark_dct,
    ensure_directories,
    list_images,
    load_or_create_key,
)


def main() -> None:
    ensure_directories()
    images = list_images(ROOT_DIR / "input")
    if not images:
        print("Chưa có ảnh. Hãy đặt ảnh cần xử lý vào thư mục input/.")
        return

    if len(images) == 1:
        source = images[0]
    else:
        print("Các ảnh tìm thấy:")
        for index, path in enumerate(images, start=1):
            print(f"  {index}. {path.name}")
        try:
            selection = int(input("Chọn số thứ tự ảnh: ").strip())
            source = images[selection - 1]
        except (ValueError, IndexError):
            print("Lựa chọn không hợp lệ.")
            return

    copyright_text = input("Nhập thông tin bản quyền: ").strip()
    if not copyright_text:
        print("Thông tin bản quyền không được để trống.")
        return

    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Không đọc được ảnh: {source.name}")
        return

    output_path = ROOT_DIR / "output" / f"{source.stem}_watermarked.png"
    try:
        key = load_or_create_key()
        encrypted = aes_encrypt(copyright_text, key)
        watermarked = embed_watermark_dct(image, encrypted)
    except (ValueError, OSError, cv2.error) as exc:
        print(f"Không thể nhúng thủy vân: {exc}")
        return

    if not cv2.imwrite(str(output_path), watermarked):
        print(f"Không thể ghi ảnh đầu ra: {output_path}")
        return
    print(f"Đã nhúng và mã hóa thủy vân: {output_path}")
    print(f"Khóa AES được lưu tại: {ROOT_DIR / 'secret.key'}")
    print("Hãy giữ bí mật và sao lưu secret.key; không thể giải mã nếu mất khóa.")


if __name__ == "__main__":
    main()
