#!/usr/bin/env python3
"""
Generate reference test data using FireRedVAD Python implementation.

This script runs all 3 models (VAD, Stream-VAD, AED) on real audio files
and saves the results as JSON reference data. The reference data is then
used to validate the C and TypeScript implementations.

Usage:
    cd fireredvad-kit
    python tests/generate_reference.py

Prerequisites:
    pip install fireredvad soundfile numpy
    # Models must be downloaded to FireRedVAD/pretrained_models/FireRedVAD/
"""

import json
import os
import sys
import shutil

import numpy as np
import soundfile as sf

# Add FireRedVAD to path
FIREREDVAD_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "FireRedVAD")
sys.path.insert(0, FIREREDVAD_ROOT)

from fireredvad import (
    FireRedVad, FireRedVadConfig,
    FireRedStreamVad, FireRedStreamVadConfig,
    FireRedAed, FireRedAedConfig,
)

# ── Audio files to test ──────────────────────────────────────────────────────

AUDIO_FILES = [
    {
        "id": "hello_en",
        "path": os.path.join(FIREREDVAD_ROOT, "assets", "hello_en.wav"),
        "description": "Short English speech (~2s)",
    },
    {
        "id": "hello_zh",
        "path": os.path.join(FIREREDVAD_ROOT, "assets", "hello_zh.wav"),
        "description": "Short Chinese speech (~2s)",
    },
    {
        "id": "event",
        "path": os.path.join(FIREREDVAD_ROOT, "assets", "event.wav"),
        "description": "Mixed audio with speech, singing, and music (~22s)",
    },
    {
        "id": "en_medium",
        "path": os.path.join(os.path.dirname(__file__), "data", "en_medium.wav"),
        "description": "English conversation (~20s, TheValley101)",
    },
    {
        "id": "zh_medium",
        "path": os.path.join(os.path.dirname(__file__), "data", "zh_medium.wav"),
        "description": "Chinese speech (~18s, DQacCB9tDaw)",
    },
]

# ── Model directories ────────────────────────────────────────────────────────

MODEL_ROOT = os.path.join(FIREREDVAD_ROOT, "pretrained_models", "FireRedVAD")
VAD_DIR = os.path.join(MODEL_ROOT, "VAD")
STREAM_VAD_DIR = os.path.join(MODEL_ROOT, "Stream-VAD")
AED_DIR = os.path.join(MODEL_ROOT, "AED")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def ensure_16k_wav(path: str) -> str:
    """Convert audio to 16kHz mono WAV if needed, return path to wav."""
    info = sf.info(path)
    if info.samplerate == 16000 and info.channels == 1:
        return path

    # Convert
    wav_path = os.path.join(OUTPUT_DIR, os.path.basename(path).rsplit(".", 1)[0] + "_16k.wav")
    if os.path.exists(wav_path):
        return wav_path

    data, sr = sf.read(path, dtype="int16")
    if len(data.shape) > 1:
        data = data[:, 0]  # take first channel
    if sr != 16000:
        # Simple resampling via ffmpeg
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", path,
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            wav_path
        ], capture_output=True)
        return wav_path
    sf.write(wav_path, data, 16000, subtype="PCM_16")
    return wav_path


def run_vad(wav_path: str) -> dict:
    """Run non-stream VAD and return results + raw probs."""
    config = FireRedVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    vad = FireRedVad.from_pretrained(VAD_DIR, config)
    result, probs = vad.detect(wav_path)

    return {
        "model": "vad",
        "config": {
            "smooth_window_size": 5,
            "speech_threshold": 0.4,
            "min_speech_frame": 20,
            "max_speech_frame": 2000,
            "min_silence_frame": 20,
        },
        "duration": result["dur"],
        "timestamps": result["timestamps"],
        "num_frames": len(probs),
        # Save first/last 10 probs + some middle probs for spot-checking
        "probs_head": probs[:10].tolist(),
        "probs_tail": probs[-10:].tolist(),
        "probs_mid": probs[len(probs)//2 : len(probs)//2 + 10].tolist() if len(probs) > 20 else [],
        "probs_mean": float(probs.mean()),
        "probs_max": float(probs.max()),
        "probs_min": float(probs.min()),
    }


def run_stream_vad(wav_path: str) -> dict:
    """Run stream VAD on full audio and return results."""
    config = FireRedStreamVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=20,
        chunk_max_frame=30000,
    )
    svad = FireRedStreamVad.from_pretrained(STREAM_VAD_DIR, config)
    frame_results, result = svad.detect_full(wav_path)

    # Extract frame-level data for validation
    confidences = [fr.raw_prob for fr in frame_results]
    speech_starts = [(fr.frame_idx, fr.speech_start_frame)
                     for fr in frame_results if fr.is_speech_start]
    speech_ends = [(fr.frame_idx, fr.speech_end_frame)
                   for fr in frame_results if fr.is_speech_end]

    return {
        "model": "stream_vad",
        "config": {
            "smooth_window_size": 5,
            "speech_threshold": 0.4,
            "pad_start_frame": 5,
            "min_speech_frame": 8,
            "max_speech_frame": 2000,
            "min_silence_frame": 20,
        },
        "duration": result["dur"],
        "timestamps": result["timestamps"],
        "num_frames": len(frame_results),
        "confidences_head": confidences[:10],
        "confidences_tail": confidences[-10:],
        "confidences_mean": float(np.mean(confidences)),
        "speech_starts": speech_starts,
        "speech_ends": speech_ends,
    }


def run_aed(wav_path: str) -> dict:
    """Run AED and return results + raw probs."""
    config = FireRedAedConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        singing_threshold=0.5,
        music_threshold=0.5,
        min_event_frame=20,
        max_event_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    aed = FireRedAed.from_pretrained(AED_DIR, config)
    result, probs = aed.detect(wav_path)

    return {
        "model": "aed",
        "config": {
            "smooth_window_size": 5,
            "speech_threshold": 0.4,
            "singing_threshold": 0.5,
            "music_threshold": 0.5,
            "min_event_frame": 20,
            "max_event_frame": 2000,
            "min_silence_frame": 20,
        },
        "duration": result["dur"],
        "event_timestamps": result["event2timestamps"],
        "event_ratios": result["event2ratio"],
        "num_frames": probs.shape[0],
        "num_classes": probs.shape[1],
        # Per-class prob stats
        "speech_probs_head": probs[:10, 0].tolist(),
        "singing_probs_head": probs[:10, 1].tolist(),
        "music_probs_head": probs[:10, 2].tolist(),
        "speech_probs_mean": float(probs[:, 0].mean()),
        "singing_probs_mean": float(probs[:, 1].mean()),
        "music_probs_mean": float(probs[:, 2].mean()),
    }


def run_fbank_reference(wav_path: str) -> dict:
    """Extract fbank features and save reference values for validation."""
    from fireredvad.core.audio_feat import AudioFeat

    # Use VAD's cmvn (all three models share the same fbank params)
    cmvn_path = os.path.join(VAD_DIR, "cmvn.ark")
    audio_feat = AudioFeat(cmvn_path)
    feat, dur = audio_feat.extract(wav_path)

    feat_np = feat.numpy()
    return {
        "num_frames": feat_np.shape[0],
        "feat_dim": feat_np.shape[1],
        "duration": dur,
        # Save first 3 frames for exact comparison
        "frames_0": feat_np[0].tolist(),
        "frames_1": feat_np[1].tolist() if feat_np.shape[0] > 1 else [],
        "frames_2": feat_np[2].tolist() if feat_np.shape[0] > 2 else [],
        # Stats
        "mean": float(feat_np.mean()),
        "std": float(feat_np.std()),
        "min": float(feat_np.min()),
        "max": float(feat_np.max()),
    }


def extract_cmvn_data() -> dict:
    """Extract CMVN means and inverse std variances for the npm package."""
    from fireredvad.core.audio_feat import CMVN

    result = {}
    for name, model_dir in [("vad", VAD_DIR), ("stream_vad", STREAM_VAD_DIR), ("aed", AED_DIR)]:
        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        if not os.path.exists(cmvn_path):
            continue
        cmvn = CMVN(cmvn_path)
        result[name] = {
            "dim": cmvn.dim,
            "means": cmvn.means.tolist(),
            "inverse_std_variances": cmvn.inverse_std_variances.tolist(),
        }
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check model availability
    for d in [VAD_DIR, STREAM_VAD_DIR, AED_DIR]:
        if not os.path.isdir(d):
            print(f"ERROR: Model directory not found: {d}")
            print("Run: modelscope download --model xukaituo/FireRedVAD --local_dir ./pretrained_models/FireRedVAD")
            sys.exit(1)

    # Extract CMVN data (needed by TypeScript package)
    print("=" * 60)
    print("Extracting CMVN data...")
    cmvn_data = extract_cmvn_data()
    cmvn_path = os.path.join(OUTPUT_DIR, "cmvn.json")
    with open(cmvn_path, "w") as f:
        json.dump(cmvn_data, f, indent=2)
    print(f"  Saved to {cmvn_path}")

    # Also save as individual binary files for C tests
    for name, data in cmvn_data.items():
        means_path = os.path.join(OUTPUT_DIR, f"cmvn_means_{name}.bin")
        istd_path = os.path.join(OUTPUT_DIR, f"cmvn_istd_{name}.bin")
        np.array(data["means"], dtype=np.float32).tofile(means_path)
        np.array(data["inverse_std_variances"], dtype=np.float32).tofile(istd_path)
        print(f"  {name}: means -> {means_path}, istd -> {istd_path}")

    # Process each audio file
    all_results = {}
    for audio in AUDIO_FILES:
        audio_id = audio["id"]
        audio_path = os.path.normpath(audio["path"])

        print(f"\n{'=' * 60}")
        print(f"Processing: {audio_id} ({audio['description']})")
        print(f"  Path: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"  SKIPPED: file not found")
            continue

        # Ensure 16kHz WAV
        wav_path = ensure_16k_wav(audio_path)
        print(f"  WAV: {wav_path}")

        # Copy wav to test data directory for C tests
        dest_wav = os.path.join(OUTPUT_DIR, f"{audio_id}.wav")
        if wav_path != dest_wav:
            shutil.copy2(wav_path, dest_wav)

        audio_results = {"id": audio_id, "description": audio["description"]}

        # Fbank reference
        print("  Running fbank...")
        audio_results["fbank"] = run_fbank_reference(wav_path)
        print(f"    {audio_results['fbank']['num_frames']} frames, dim={audio_results['fbank']['feat_dim']}")

        # VAD
        print("  Running VAD...")
        audio_results["vad"] = run_vad(wav_path)
        print(f"    duration={audio_results['vad']['duration']}s, "
              f"segments={len(audio_results['vad']['timestamps'])}")

        # Stream VAD
        print("  Running Stream VAD...")
        audio_results["stream_vad"] = run_stream_vad(wav_path)
        print(f"    duration={audio_results['stream_vad']['duration']}s, "
              f"segments={len(audio_results['stream_vad']['timestamps'])}")

        # AED
        print("  Running AED...")
        audio_results["aed"] = run_aed(wav_path)
        print(f"    duration={audio_results['aed']['duration']}s")
        for event, ts in audio_results["aed"]["event_timestamps"].items():
            print(f"    {event}: {len(ts)} segments, ratio={audio_results['aed']['event_ratios'][event]}")

        all_results[audio_id] = audio_results

    # Save all results
    output_path = os.path.join(OUTPUT_DIR, "reference_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"All reference data saved to: {output_path}")
    print(f"Audio files: {len(all_results)}")
    print(f"Models tested: VAD, Stream-VAD, AED")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    for audio_id, res in all_results.items():
        vad_segs = len(res["vad"]["timestamps"])
        svad_segs = len(res["stream_vad"]["timestamps"])
        aed_events = sum(len(v) for v in res["aed"]["event_timestamps"].values())
        print(f"  {audio_id}: VAD={vad_segs} segs, StreamVAD={svad_segs} segs, AED={aed_events} events")


if __name__ == "__main__":
    main()
