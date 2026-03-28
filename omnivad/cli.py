"""OmniVAD command-line interface — audio → TextGrid."""

import argparse
import sys
import time


def write_textgrid(path, duration, tiers):
    """Write a Praat TextGrid file.

    tiers: list of (name, segments) where segments = [(start, end, label), ...]
    """
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


def main():
    parser = argparse.ArgumentParser(
        prog="omnivad",
        description="OmniVAD — Voice Activity Detection & Audio Event Detection → TextGrid",
    )
    parser.add_argument("audio", help="Input audio file (16kHz mono WAV)")
    parser.add_argument("-o", "--output", help="Output TextGrid path (default: <audio>.TextGrid)")
    parser.add_argument(
        "-m",
        "--mode",
        default="all",
        help="Detection mode: vad, aed, or all (default: all)",
    )
    args = parser.parse_args()
    args.mode = args.mode.lower()
    if args.mode not in ("vad", "aed", "all"):
        parser.error(f"invalid mode: {args.mode} (choose from vad, aed, all)")

    from omnivad import OmniAED, OmniVAD

    output_path = args.output or args.audio.rsplit(".", 1)[0] + ".TextGrid"

    tiers = []
    duration = None

    # -- Load models --
    t_init = time.perf_counter()
    vad = OmniVAD() if args.mode in ("vad", "all") else None
    aed = OmniAED() if args.mode in ("aed", "all") else None
    init_elapsed = time.perf_counter() - t_init

    # -- VAD --
    if vad:
        t1 = time.perf_counter()
        result = vad.detect(args.audio)
        vad_elapsed = time.perf_counter() - t1
        duration = result["duration"]
        tiers.append(("VAD", [(s, e, "speech") for s, e in result["timestamps"]]))
        vad_rtf = vad_elapsed / duration
        print(f"VAD: {len(result['timestamps'])} segments, {vad_elapsed:.3f}s, RTF={vad_rtf:.4f}")

    # -- AED --
    if aed:
        t2 = time.perf_counter()
        result = aed.detect(args.audio)
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

    write_textgrid(output_path, duration, tiers)
    total = time.perf_counter() - t_init
    compute = total - init_elapsed
    print(f"\nAudio: {duration:.3f}s | Init: {init_elapsed:.3f}s | Compute: {compute:.3f}s | Total: {total:.3f}s")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
