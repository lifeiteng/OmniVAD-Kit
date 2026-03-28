"""OmniVAD command-line interface — audio → TextGrid / JSON / SRT / VTT."""

import argparse
import json
import os
import sys
import time


# -- Output formatters --


def _fmt_ts(seconds):
    """Format seconds as HH:MM:SS.mmm for SRT/VTT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_textgrid(path, duration, tiers):
    """Write a Praat TextGrid file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write(f"xmin = 0\nxmax = {duration}\n")
        f.write("tiers? <exists>\n")
        f.write(f"size = {len(tiers)}\n")
        f.write("item []:\n")

        for i, (name, segments) in enumerate(tiers, 1):
            f.write(f"    item [{i}]:\n")
            f.write(f'        class = "IntervalTier"\n')
            f.write(f'        name = "{name}"\n')
            f.write(f"        xmin = 0\n")
            f.write(f"        xmax = {duration}\n")

            intervals = []
            prev_end = 0.0
            for start, end, label in sorted(segments):
                if start > prev_end + 0.001:
                    intervals.append((prev_end, start, ""))
                intervals.append((start, end, label))
                prev_end = end
            if prev_end < duration - 0.001:
                intervals.append((prev_end, duration, ""))
            if not intervals:
                intervals = [(0, duration, "")]

            f.write(f"        intervals: size = {len(intervals)}\n")
            for j, (s, e, label) in enumerate(intervals, 1):
                f.write(f"        intervals [{j}]:\n")
                f.write(f"            xmin = {s}\n")
                f.write(f"            xmax = {e}\n")
                f.write(f'            text = "{label}"\n')


def write_json(path, duration, tiers):
    """Write detection results as JSON."""
    data = {"duration": duration, "tiers": {}}
    for name, segments in tiers:
        data["tiers"][name] = [{"start": s, "end": e, "label": label} for s, e, label in segments]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_srt(path, _duration, tiers):
    """Write detection results as SRT (all tiers merged, label as text)."""
    all_segs = []
    for _name, segments in tiers:
        for s, e, label in segments:
            all_segs.append((s, e, label))
    all_segs.sort()

    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, label) in enumerate(all_segs, 1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_ts(s).replace('.', ',')} --> {_fmt_ts(e).replace('.', ',')}\n")
            f.write(f"{label}\n\n")


def write_vtt(path, _duration, tiers):
    """Write detection results as WebVTT."""
    all_segs = []
    for _name, segments in tiers:
        for s, e, label in segments:
            all_segs.append((s, e, label))
    all_segs.sort()

    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s, e, label in all_segs:
            f.write(f"{_fmt_ts(s)} --> {_fmt_ts(e)}\n")
            f.write(f"{label}\n\n")


WRITERS = {
    ".textgrid": write_textgrid,
    ".json": write_json,
    ".srt": write_srt,
    ".vtt": write_vtt,
}

FORMAT_EXT = {
    "textgrid": ".TextGrid",
    "json": ".json",
    "srt": ".srt",
    "vtt": ".vtt",
}


def main():
    parser = argparse.ArgumentParser(
        prog="omnivad",
        description="OmniVAD — Voice Activity Detection & Audio Event Detection",
    )
    parser.add_argument("audio", help="Input audio file (16kHz mono WAV)")
    parser.add_argument("-o", "--output", help="Output path (default: <audio>.<format>)")
    parser.add_argument(
        "-f",
        "--format",
        choices=["textgrid", "json", "srt", "vtt"],
        help="Output format (default: inferred from -o extension, or textgrid)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="all",
        help="Detection mode: vad, aed, or all (default: all)",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=0,
        metavar="SEC",
        help="Process in chunks of SEC seconds (for large audio, e.g. 600)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Overlap between chunks in seconds (default: 2.0)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel threads for chunked processing (default: 4)",
    )
    args = parser.parse_args()
    args.mode = args.mode.lower()
    if args.mode not in ("vad", "aed", "all"):
        parser.error(f"invalid mode: {args.mode} (choose from vad, aed, all)")

    # Resolve output format and path
    if args.format:
        ext = FORMAT_EXT[args.format]
    elif args.output:
        ext = os.path.splitext(args.output)[1]
        if ext.lower() not in WRITERS:
            ext = ".TextGrid"
    else:
        ext = ".TextGrid"

    output_path = args.output or args.audio.rsplit(".", 1)[0] + ext
    writer = WRITERS.get(os.path.splitext(output_path)[1].lower(), write_textgrid)

    from omnivad import OmniAED, OmniVAD

    tiers = []
    duration = None
    detect_kwargs = {}
    if args.chunk > 0:
        detect_kwargs = {"chunk_seconds": args.chunk, "overlap_seconds": args.overlap, "workers": args.workers}

    # -- Load models --
    t_init = time.perf_counter()
    vad = OmniVAD() if args.mode in ("vad", "all") else None
    aed = OmniAED() if args.mode in ("aed", "all") else None
    init_elapsed = time.perf_counter() - t_init

    # -- VAD --
    if vad:
        t1 = time.perf_counter()
        result = vad.detect(args.audio, **detect_kwargs)
        vad_elapsed = time.perf_counter() - t1
        duration = result["duration"]
        tiers.append(("VAD", [(s, e, "speech") for s, e in result["timestamps"]]))
        vad_rtf = vad_elapsed / duration
        print(f"VAD: {len(result['timestamps'])} segments, {vad_elapsed:.3f}s, RTF={vad_rtf:.4f}")

    # -- AED --
    if aed:
        t2 = time.perf_counter()
        result = aed.detect(args.audio, **detect_kwargs)
        aed_elapsed = time.perf_counter() - t2
        duration = result["duration"]
        for cls in ("speech", "singing", "music"):
            segs = result["events"].get(cls, [])
            tiers.append((f"AED-{cls}", [(s, e, cls) for s, e in segs]))
        total_events = sum(len(result["events"].get(c, [])) for c in ("speech", "singing", "music"))
        aed_rtf = aed_elapsed / duration
        print(f"AED: {total_events} events, {aed_elapsed:.3f}s, RTF={aed_rtf:.4f}")

    if duration is None:
        print("Error: no model was run", file=sys.stderr)
        sys.exit(1)

    writer(output_path, duration, tiers)
    total = time.perf_counter() - t_init
    compute = total - init_elapsed
    chunk_info = f" | Chunk: {args.chunk}s/{args.overlap}s overlap" if args.chunk > 0 else ""
    print(f"\nAudio: {duration:.3f}s | Init: {init_elapsed:.3f}s | Compute: {compute:.3f}s | Total: {total:.3f}s{chunk_info}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
