#!/usr/bin/env python3
"""Fetch the pinned Qwen2.5-7B-Instruct revision, check every file's SHA256, and
write the manifest that run.py reads (application_model.json).

    python3 prepare_model.py MANIFEST [--cache-dir DIR]   # download (about 15 GB)
    python3 prepare_model.py MANIFEST --snapshot DIR      # an existing local copy

The expected hashes are those of the files the published runs loaded
(model_files.json). The manifest also records each file's size and mtime;
run.py refuses to start if either changed after hashing.
"""
import argparse, hashlib, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
expected = json.loads((HERE / "model_files.json").read_text())
config = json.loads((HERE / "application.json").read_text())
assert (expected["model_id"], expected["revision"]) == (config["model_id"], config["revision"])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("manifest", type=pathlib.Path, help="output path, e.g. model/application_model.json")
parser.add_argument("--cache-dir", help="Hugging Face cache directory for the download")
parser.add_argument("--snapshot", type=pathlib.Path, help="use this local snapshot directory instead of downloading")
args = parser.parse_args()

if args.snapshot:
    snapshot = args.snapshot.resolve()
else:
    from huggingface_hub import snapshot_download
    snapshot = pathlib.Path(snapshot_download(config["model_id"], revision=config["revision"], cache_dir=args.cache_dir))

records = []
for item in expected["files"]:
    f = snapshot / item["name"]
    before = f.stat()
    h = hashlib.sha256()
    with f.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = f.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"{f} changed while hashing"
    assert (after.st_size, h.hexdigest()) == (item["bytes"], item["sha256"]), f"{item['name']} differs from the measured revision"
    records.append({"name": item["name"], "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": h.hexdigest()})
    print(item["name"], h.hexdigest(), flush=True)

args.manifest.parent.mkdir(parents=True, exist_ok=True)
args.manifest.write_text(json.dumps({"model_id": config["model_id"], "revision": config["revision"],
                                     "snapshot": str(snapshot), "files": records}, indent=2) + "\n")
print("verified snapshot:", snapshot, "->", args.manifest)
