#!/usr/bin/env python3
"""
Pack ncnn model files (param + bin + cmvn) into a single .omnivad bundle.

Bundle format:
    [Magic: "OVAD"]         4 bytes
    [Version: uint32]       4 bytes  (currently 1)
    [Param size: uint32]    4 bytes
    [Bin size: uint32]      4 bytes
    [CMVN means size: uint32] 4 bytes
    [CMVN istd size: uint32]  4 bytes
    [Param data ...]        param_size bytes
    [Bin data ...]          bin_size bytes
    [CMVN means data ...]   cmvn_means_size bytes
    [CMVN istd data ...]    cmvn_istd_size bytes

Usage:
    python scripts/pack_model.py --param model.ncnn.param --bin model.ncnn.bin \
        --cmvn-means cmvn_means.bin --cmvn-istd cmvn_istd.bin -o model.omnivad

    # Or pack all 3 models at once:
    python scripts/pack_model.py --all
"""

import argparse
import os
import struct
import sys

MAGIC = b"OVAD"
VERSION = 1
HEADER_SIZE = 4 + 4 + 4 * 4  # magic + version + 4 sizes = 24 bytes

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def pack(param_path, bin_path, cmvn_means_path, cmvn_istd_path, output_path):
    param_data = open(param_path, "rb").read()
    bin_data = open(bin_path, "rb").read()
    cmvn_means = open(cmvn_means_path, "rb").read()
    cmvn_istd = open(cmvn_istd_path, "rb").read()

    header = MAGIC + struct.pack("<I", VERSION)
    header += struct.pack("<IIII", len(param_data), len(bin_data), len(cmvn_means), len(cmvn_istd))

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(param_data)
        f.write(bin_data)
        f.write(cmvn_means)
        f.write(cmvn_istd)

    total = os.path.getsize(output_path)
    print(
        f"  {output_path}: {total:,} bytes "
        f"(param={len(param_data):,}, bin={len(bin_data):,}, "
        f"cmvn={len(cmvn_means) + len(cmvn_istd):,})"
    )


def pack_all():
    fireredvad = os.path.join(PROJECT_DIR, "..", "FireRedVAD")
    onnx_dir = os.path.join(fireredvad, "pretrained_models", "onnx_models")
    stream_dir = os.path.join(fireredvad, "runtime", "convert", "out")
    data_dir = os.path.join(PROJECT_DIR, "tests", "data")
    out_dir = os.path.join(PROJECT_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)

    models = [
        {
            "name": "vad",
            "param": os.path.join(onnx_dir, "fireredvad_vad.ncnn.param"),
            "bin": os.path.join(onnx_dir, "fireredvad_vad.ncnn.bin"),
            "means": os.path.join(data_dir, "cmvn_means_vad.bin"),
            "istd": os.path.join(data_dir, "cmvn_istd_vad.bin"),
        },
        {
            "name": "aed",
            "param": os.path.join(onnx_dir, "fireredvad_aed.ncnn.param"),
            "bin": os.path.join(onnx_dir, "fireredvad_aed.ncnn.bin"),
            "means": os.path.join(data_dir, "cmvn_means_aed.bin"),
            "istd": os.path.join(data_dir, "cmvn_istd_aed.bin"),
        },
        {
            "name": "stream-vad",
            "param": os.path.join(stream_dir, "stream_packed.ncnn.param"),
            "bin": os.path.join(stream_dir, "stream_packed.ncnn.bin"),
            "means": os.path.join(stream_dir, "cmvn_means_stream.bin"),
            "istd": os.path.join(stream_dir, "cmvn_istd_stream.bin"),
        },
    ]

    for m in models:
        out_path = os.path.join(out_dir, f"{m['name']}.omnivad")
        for key in ["param", "bin", "means", "istd"]:
            if not os.path.exists(m[key]):
                print(f"  SKIP {m['name']}: {m[key]} not found")
                break
        else:
            pack(m["param"], m["bin"], m["means"], m["istd"], out_path)


def main():
    parser = argparse.ArgumentParser(description="Pack ncnn model into .omnivad bundle")
    parser.add_argument("--param", help="ncnn .param file")
    parser.add_argument("--bin", help="ncnn .bin file")
    parser.add_argument("--cmvn-means", help="CMVN means binary file")
    parser.add_argument("--cmvn-istd", help="CMVN inverse std binary file")
    parser.add_argument("-o", "--output", help="Output .omnivad file")
    parser.add_argument("--all", action="store_true", help="Pack all 3 models")
    args = parser.parse_args()

    if args.all:
        pack_all()
    elif args.param and args.bin and args.cmvn_means and args.cmvn_istd and args.output:
        pack(args.param, args.bin, args.cmvn_means, args.cmvn_istd, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
