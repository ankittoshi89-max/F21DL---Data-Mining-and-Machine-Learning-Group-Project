
import argparse
import time
import urllib.request
import zipfile

from config import RAW_DIR, RGB_URL, MS_URL

CHUNK = 1024 * 1024      # 1 MB at a time
MAX_TRIES = 20


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def download(url, dest):
    if dest.exists():
        print(f"{dest.name} already downloaded, skipping")
        return

    tmp = dest.with_suffix(".part")
    total = remote_size(url)
    print(f"Downloading {dest.name} ({total / 1e6:.0f} MB)")

    for attempt in range(1, MAX_TRIES + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total and have >= total:
            break
        try:
            # ask the server only for the bytes we don't have yet
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:      # server ignored the range -> start again
                    have = 0
                mode = "ab" if have else "wb"
                with open(tmp, mode) as f:
                    while True:
                        block = r.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        have += len(block)
                        if total:
                            print(f"\r  {dest.name}: {have / total * 100:5.1f}%  ({have / 1e6:.0f}/{total / 1e6:.0f} MB)", end="")
            if not total or have >= total:
                break
        except Exception as e:
            print(f"\n  connection problem ({e.__class__.__name__}), retry {attempt}/{MAX_TRIES} in 5 s ...")
            time.sleep(5)
    else:
        raise SystemExit("Download kept failing - run the script again later, it will resume.")

    print()
    if total and tmp.stat().st_size != total:
        raise SystemExit("Downloaded size does not match - run the script again to resume.")
    tmp.rename(dest)


def unzip(zip_path, out_dir):
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{out_dir.name} already unzipped, skipping")
        return
    print(f"Unzipping {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    except zipfile.BadZipFile:
        zip_path.unlink()
        raise SystemExit(f"{zip_path.name} was corrupted and has been deleted - run the script again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-only", action="store_true")
    args = parser.parse_args()

    jobs = [("EuroSAT_RGB", RGB_URL)]
    if not args.rgb_only:
        jobs.append(("EuroSAT_MS", MS_URL))

    for name, url in jobs:
        zip_path = RAW_DIR / f"{name}.zip"
        download(url, zip_path)
        unzip(zip_path, RAW_DIR / name)

    print("Done. Data is in", RAW_DIR)