import argparse
import cv2
import numpy as np
import librosa
import mediapipe as mp
from scipy.signal import find_peaks
from pathlib import Path
from typing import List, Tuple
import time


def detect_voice_peaks(
    video_path: str,
    sample_rate: int = 16000,
    frame_length: int = 2048,
    hop_length: int = 512,
    min_pitch_hz: float = 60.0,
    max_pitch_hz: float = 450.0,
    z_threshold: float = 1.25,
    min_distance_s: float = 0.6,
) -> List[Tuple[float, float]]:
    """Return (timestamp_sec, salience) for emphasized voice moments.

    Salience is the combined z-score of RMS and pitch where voiced, or RMS z elsewhere.
    """
    y, sr = librosa.load(video_path, sr=sample_rate, mono=True)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    f0 = librosa.yin(
        y=y,
        fmin=min_pitch_hz,
        fmax=max_pitch_hz,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    def zscore(arr: np.ndarray) -> np.ndarray:
        mean_val = np.nanmean(arr)
        std_val = np.nanstd(arr) + 1e-8
        return (arr - mean_val) / std_val

    rms_z = zscore(rms)
    f0_z = zscore(f0)

    voiced_mask = ~np.isnan(f0)

    combined = np.zeros_like(rms_z)
    combined[voiced_mask] = 0.7 * rms_z[voiced_mask] + 0.3 * f0_z[voiced_mask]
    combined[~voiced_mask] = rms_z[~voiced_mask]

    min_distance_frames = max(1, int(min_distance_s / (hop_length / sr)))
    peaks, props = find_peaks(combined, height=z_threshold, distance=min_distance_frames)

    peak_times = times[peaks].tolist()
    peak_scores = combined[peaks].tolist()
    return list(zip(peak_times, peak_scores))


def detect_gesture_peaks(
    video_path: str,
    movement_prominence: float = 0.5,
    min_distance_s: float = 0.6,
    frame_stride: int = 1,
    ema_alpha: float = 0.2,
    pose_model_complexity: int = 0,
    progress: bool = False,
    log_every_sec: float = 2.0,
) -> List[Tuple[float, float]]:
    """Return (timestamp_sec, salience) for unusually high body movement.

    Salience is robust z-score of smoothed landmark velocity.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    mp_pose = mp.solutions.pose

    movement_series: list[float] = []

    last_log = time.time()

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(pose_model_complexity),
            enable_segmentation=False,
            smooth_landmarks=True,
        ) as pose:
            prev_coords = None
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_stride > 1 and (frame_idx % frame_stride) != 0:
                    frame_idx += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    indices = [
                        mp_pose.PoseLandmark.LEFT_SHOULDER,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        mp_pose.PoseLandmark.LEFT_ELBOW,
                        mp_pose.PoseLandmark.RIGHT_ELBOW,
                        mp_pose.PoseLandmark.LEFT_WRIST,
                        mp_pose.PoseLandmark.RIGHT_WRIST,
                    ]
                    coords = np.array([[lm[i].x, lm[i].y] for i in indices], dtype=np.float32)

                    if prev_coords is not None:
                        vel = np.linalg.norm(coords - prev_coords, axis=1).mean()
                    else:
                        vel = 0.0
                    prev_coords = coords
                else:
                    vel = 0.0

                if movement_series:
                    smoothed = ema_alpha * vel + (1.0 - ema_alpha) * movement_series[-1]
                else:
                    smoothed = vel
                movement_series.append(smoothed)

                frame_idx += 1

                if progress and (time.time() - last_log) >= log_every_sec:
                    if total_frames > 0:
                        pct = 100.0 * (frame_idx / total_frames)
                        print(f"Pose processing: {frame_idx}/{total_frames} frames ({pct:.1f}%)")
                    else:
                        print(f"Pose processing: {frame_idx} frames")
                    last_log = time.time()
    finally:
        cap.release()

    movement = np.asarray(movement_series, dtype=np.float32)

    def robust_norm(x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-8
        return (x - med) / (1.4826 * mad)

    movement_z = robust_norm(movement)

    distance_frames = max(1, int(min_distance_s * fps / max(1, frame_stride)))

    peaks, props = find_peaks(movement_z, prominence=movement_prominence, distance=distance_frames)

    times_arr = (peaks * frame_stride) / fps
    scores = movement_z[peaks]
    return list(zip(times_arr.astype(float).tolist(), scores.astype(float).tolist()))


def merge_and_deduplicate_timestamps(
    a: List[Tuple[float, float]],
    b: List[Tuple[float, float]],
    min_separation_s: float = 0.5,
) -> List[Tuple[float, float]]:
    """Merge two lists of (time, score) and remove near-duplicates by keeping the max score in each neighborhood."""
    all_points = sorted(a + b, key=lambda x: x[0])
    if not all_points:
        return []

    merged: List[Tuple[float, float]] = []
    current_time, current_score = all_points[0]
    for t, s in all_points[1:]:
        if t - current_time < min_separation_s:
            if s > current_score:
                current_time, current_score = t, s
        else:
            merged.append((current_time, current_score))
            current_time, current_score = t, s
    merged.append((current_time, current_score))
    return merged


def save_keyframes(video_path: str, seconds_list: list[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    for sec in seconds_list:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(out_dir / f"frame_{sec:04d}.jpg"), frame)
    cap.release()


def get_video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 1e-6:
        fps = 30.0
    return float(frame_count / fps) if frame_count > 0 else 0.0


def choose_top_seconds(points: List[Tuple[float, float]], duration_s: float, max_per_minute: float) -> List[int]:
    """Quantize candidate timestamps to whole seconds, keep highest score per second, then cap to top-N by score.

    Returns sorted integer seconds.
    """
    if not points:
        return []

    duration_sec_int = max(0, int(np.floor(duration_s)))
    # best score per second
    best_by_sec: dict[int, float] = {}
    for t, s in points:
        sec = int(round(t))
        if sec < 0 or sec >= duration_sec_int:
            continue
        prev = best_by_sec.get(sec)
        if (prev is None) or (s > prev):
            best_by_sec[sec] = float(s)

    # Cap N = ceil(minutes * max_per_minute)
    minutes = max(1e-6, duration_s / 60.0)
    cap_n = int(np.ceil(minutes * max_per_minute))

    # Sort by score desc, take top cap_n, then sort ascending by second
    top_secs = sorted(best_by_sec.items(), key=lambda kv: kv[1], reverse=True)[:cap_n]
    top_secs_sorted = sorted([sec for sec, _ in top_secs])
    return top_secs_sorted


def main():
    parser = argparse.ArgumentParser(description="Extract key timestamps from a speaking video.")
    parser.add_argument("video", type=str, help="Path to input .mp4 video")
    parser.add_argument("--save-frames", action="store_true", help="Save key frames as JPEGs")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory root (defaults to ilak/outputs/<video_stem>)")
    parser.add_argument("--audio-thresh", type=float, default=1.25, help="Audio z-score threshold")
    parser.add_argument("--gesture-prom", type=float, default=0.5, help="Gesture prominence threshold (robust z)")
    parser.add_argument("--min-gap", type=float, default=0.6, help="Minimum gap between peaks in seconds")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame for speed")
    parser.add_argument("--max-per-minute", type=float, default=10.0, help="Cap number of keyframes to this rate per minute")
    parser.add_argument("--pose-complexity", type=int, default=0, choices=[0, 1, 2], help="MediaPipe pose model complexity (0 fastest)")
    parser.add_argument("--progress", action="store_true", help="Print pose processing progress periodically")
    args = parser.parse_args()

    video_path = args.video
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    print(f"Analyzing video: {video_file}")

    voice_points = detect_voice_peaks(
        str(video_file), z_threshold=args.audio_thresh, min_distance_s=args.min_gap
    )
    print(f"Detected {len(voice_points)} voice peaks")

    gesture_points = detect_gesture_peaks(
        str(video_file),
        movement_prominence=args.gesture_prom,
        min_distance_s=args.min_gap,
        frame_stride=args.stride,
        ema_alpha=0.2,
        pose_model_complexity=args.pose_complexity,
        progress=args.progress,
    )
    print(f"Detected {len(gesture_points)} gesture peaks")

    merged_points = merge_and_deduplicate_timestamps(voice_points, gesture_points, min_separation_s=args.min_gap)

    duration_s = get_video_duration_seconds(str(video_file))

    key_seconds = choose_top_seconds(merged_points, duration_s, args.max_per_minute)

    print("\n=== Keyframe Seconds ===")
    print(key_seconds)

    if args.save_frames and key_seconds:
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            script_dir = Path(__file__).parent
            out_dir = script_dir / "outputs" / video_file.stem
        save_keyframes(str(video_file), key_seconds, out_dir)
        print(f"\nFrames saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
