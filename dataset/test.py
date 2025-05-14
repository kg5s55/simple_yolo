import os
import glob
from pathlib import Path

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes


def get_img_files(img_path):
    try:
        f = []  # image files
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in t]
            else:
                raise FileNotFoundError(f"{p} does not exist")
        im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
        assert im_files, f"No images found in {img_path}. "
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from {img_path}\n") from e

    return im_files


if __name__ == "__main__":
    img_path0 = r"F:\dataset\labeled_tt100k\images"
    x = r"F:\dataset\labeled_tt100k\images\train\1.png"
    # get_img_files(img_path0)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    print(sa)
    print(x.rsplit(sa, 1))
    print(os.cpu_count())

