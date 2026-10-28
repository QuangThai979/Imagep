"""Extract and decrypt copyright text from images in input_check/."""

from datetime import datetime

import cv2

from watermark import (
    ROOT_DIR,
    aes_decrypt,
    ensure_directories,
    extract_watermark_dct,
    list_images,
    load_or_create_key,
)


def main() -> None:
    ensure_directories()
    images = list_images(ROOT_DIR / "input_check")
    result_path = ROOT_DIR / "output_check" / "result.txt"
    results = [f"Kết quả kiểm tra - {datetime.now().astimezone().isoformat(timespec='seconds')}"]

    if not images:
        results.append("THẤT BẠI: Không tìm thấy ảnh trong thư mục input_check/.")
    else:
        try:
            key = load_or_create_key()
        except (ValueError, OSError) as exc:
            results.append(f"THẤT BẠI: Không thể tải khóa AES: {exc}")
        else:
            for image_path in images:
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    results.append(f"[{image_path.name}] THẤT BẠI: Không đọc được ảnh.")
                    continue
                try:
                    encrypted = extract_watermark_dct(image)
                    copyright_text = aes_decrypt(encrypted, key)
                except (ValueError, OSError, cv2.error) as exc:
                    results.append(f"[{image_path.name}] THẤT BẠI: {exc}")
                else:
                    results.append(
                        f"[{image_path.name}] THÀNH CÔNG: {copyright_text}"
                    )

    result_path.write_text("\n".join(results) + "\n", encoding="utf-8")
    print("\n".join(results))
    print(f"\nĐã lưu kết quả tại: {result_path}")


if __name__ == "__main__":
    main()
