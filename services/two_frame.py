import json
import subprocess
import shutil
from pathlib import Path


def create_adaptive_face_short(
    input_path: Path,
    output_path: Path,
    fallback_output: Path,
    mode: str = "auto_face",
) -> bool:
    """Render a portrait video whose panel count follows visible faces.

    Face detection is sampled at five frames and held briefly between samples.
    This avoids the distracting panel flicker caused by running a Haar cascade
    independently on every frame, while still adapting as a guest enters or
    leaves the source shot. The original audio is reattached after OpenCV
    produces the dynamic video frames.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[two_frame] opencv/numpy unavailable; using regular portrait crop")
        from .cropper import crop_to_vertical
        crop_to_vertical(input_path, fallback_output)
        return False

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        from .cropper import crop_to_vertical
        crop_to_vertical(input_path, fallback_output)
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        from .cropper import crop_to_vertical
        crop_to_vertical(input_path, fallback_output)
        return False

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        cap.release()
        from .cropper import crop_to_vertical
        crop_to_vertical(input_path, fallback_output)
        return False

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        silent_path = Path(temp_dir) / "adaptive_silent.mp4"
        writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1080, 1920))
        if not writer.isOpened():
            cap.release()
            from .cropper import crop_to_vertical
            crop_to_vertical(input_path, fallback_output)
            return False

        frame_index, detected_frames, missing_frames = 0, 0, 0
        faces: list[tuple[int, int, int, int]] = []
        scan_every = max(1, round(fps / 6))  # roughly six detector passes/sec
        hold_for = max(1, round(fps * 0.8))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % scan_every == 0:
                gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                found = cascade.detectMultiScale(
                    gray, scaleFactor=1.08, minNeighbors=4,
                    minSize=(max(28, frame_w // 32), max(28, frame_h // 32)),
                )
                candidates = sorted((tuple(map(int, face)) for face in found), key=lambda f: f[2] * f[3], reverse=True)[:3]
                if candidates:
                    # Sorting left-to-right makes panel placement stable when faces are similar sizes.
                    faces = sorted(candidates, key=lambda f: f[0] + f[2] / 2)
                    detected_frames += 1
                    missing_frames = 0
                else:
                    missing_frames += scan_every
                    if missing_frames > hold_for:
                        faces = []

            count = _panel_count(mode, len(faces))
            writer.write(_compose_face_frame(frame, faces[:count], count, cv2, np))
            frame_index += 1
        writer.release()
        cap.release()
        if not frame_index or not detected_frames:
            from .cropper import crop_to_vertical
            crop_to_vertical(input_path, fallback_output)
            return False
        _attach_original_audio(silent_path, input_path, output_path)
    print(f"[two_frame] adaptive {mode} layout rendered from {detected_frames} face samples")
    return True


def _panel_count(mode: str, available: int) -> int:
    if mode == "two_frame":
        return 2 if available >= 2 else 1
    # Both multi-face and auto-face use the observed count; auto-face simply
    # makes the time-varying behavior explicit in the product UI.
    return max(1, min(3, available))


def _compose_face_frame(frame, faces, count: int, cv2, np):
    canvas = np.zeros((1920, 1080, 3), dtype=np.uint8)
    panel_h = 1920 // count
    for index in range(count):
        face = faces[index] if index < len(faces) else None
        panel = _crop_face_to_panel(frame, face, 1080, panel_h, cv2)
        y1 = index * panel_h
        y2 = 1920 if index == count - 1 else y1 + panel_h
        canvas[y1:y2] = panel[: y2 - y1]
        if index:
            cv2.line(canvas, (0, y1), (1080, y1), (0, 0, 0), 8)
    return canvas


def _crop_face_to_panel(frame, face, target_w: int, target_h: int, cv2):
    height, width = frame.shape[:2]
    if face:
        x, y, w, h = face
        center_x, center_y = x + w / 2, y + h / 2
        crop_h = max(h * 2.7, target_h * min(width / target_w, height / target_h))
    else:
        center_x, center_y, crop_h = width / 2, height / 2, height
    crop_w = crop_h * target_w / target_h
    # Expand until the requested portrait panel fits source bounds.
    scale = max(crop_w / width, crop_h / height, 0.01)
    if scale > 1:
        crop_w, crop_h = width, height
    left = max(0, min(int(center_x - crop_w / 2), max(0, int(width - crop_w))))
    top = max(0, min(int(center_y - crop_h / 2), max(0, int(height - crop_h))))
    right, bottom = max(left + 1, int(left + crop_w)), max(top + 1, int(top + crop_h))
    crop = frame[top:bottom, left:right]
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def _attach_original_audio(silent_path: Path, original_path: Path, output_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-i", str(silent_path), "-i", str(original_path),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(output_path),
    ], check=True, capture_output=True)


def detect_two_faces(video_path: Path) -> list[dict] | None:
    try:
        import cv2
    except ImportError:
        print("[two_frame] opencv not installed — pip install opencv-python")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[two_frame] could not open video for face detection")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    scan_duration = 30  # seconds
    max_frames = int(fps * scan_duration)
    sample_every = max(1, int(fps))  # sample 1 frame per second

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Accumulate all detected face regions across sampled frames
    face_regions: list[dict] = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )
            for (x, y, w, h) in faces:
                # Only add if this is a new distinct face region
                is_new = True
                for existing in face_regions:
                    overlap_x = abs(existing["x"] - x) < existing["w"] * 0.5
                    overlap_y = abs(existing["y"] - y) < existing["h"] * 0.5
                    if overlap_x and overlap_y:
                        is_new = False
                        break
                if is_new:
                    face_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                    })

        frame_idx += 1

    cap.release()

    if len(face_regions) < 2:
        print(f"[two_frame] only {len(face_regions)} face(s) detected — need 2")
        return None

    # Return the two most prominent faces (largest bounding boxes)
    face_regions.sort(key=lambda f: f["w"] * f["h"], reverse=True)
    return face_regions[:2]


def crop_face_region(
    input_path: Path,
    output_path: Path,
    face: dict,
    target_w: int = 1080,
    target_h: int = 960,
    padding: float = 0.4,
) -> None:
    x = face["x"]
    y = face["y"]
    w = face["w"]
    h = face["h"]

    # Add padding around the face so we see shoulders and background
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    # Get actual video dimensions first
    video_w, video_h = _get_dimensions(input_path)

    crop_x = max(0, x - pad_x)
    crop_y = max(0, y - pad_y)
    crop_w = min(video_w - crop_x, w + pad_x * 2)
    crop_h = min(video_h - crop_y, h + pad_y * 2)

    # Ensure even numbers (FFmpeg requirement)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={target_w}:{target_h}:flags=lanczos,"
        "format=yuv420p",
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "medium",
        "-an",  # no audio in individual face crops
        str(output_path),
    ], check=True, capture_output=True)


def stack_two_faces(
    top_path: Path,
    bottom_path: Path,
    original_path: Path,
    output_path: Path,
    target_w: int = 1080,
    target_h: int = 1920,
) -> None:
    half_h = target_h // 2  # 960

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(top_path),
        "-i", str(bottom_path),
        "-i", str(original_path),
        "-filter_complex",
        f"[0:v]scale={target_w}:{half_h}[top];"
        f"[1:v]scale={target_w}:{half_h}[bottom];"
        f"[top][bottom]vstack=inputs=2[stacked]",
        "-map", "[stacked]",
        "-map", "2:a",
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ], check=True, capture_output=True)


def create_two_frame_short(
    input_path: Path,
    output_path: Path,
    fallback_output: Path,
) -> bool:
    """
    Master function. Detects two faces, crops each, stacks vertically.
    Returns True if two-frame was created, False if fell back to single frame.
    """
    print("[two_frame] starting face detection")
    faces = detect_two_faces(input_path)

    if faces is None:
        print("[two_frame] could not find 2 faces — falling back to single frame crop")
        # Fall back to regular 9:16 crop
        from .cropper import crop_to_vertical
        crop_to_vertical(input_path, fallback_output)
        return False

    face_a = faces[0]
    face_b = faces[1]
    print(f"[two_frame] face A: {face_a}")
    print(f"[two_frame] face B: {face_b}")

    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as tmp:
        tmp = P(tmp)
        top_path    = tmp / "face_top.mp4"
        bottom_path = tmp / "face_bottom.mp4"

        print("[two_frame] cropping face A (top)")
        crop_face_region(input_path, top_path, face_a)

        print("[two_frame] cropping face B (bottom)")
        crop_face_region(input_path, bottom_path, face_b)

        print("[two_frame] stacking faces vertically")
        stack_two_faces(top_path, bottom_path, input_path, output_path)

    print("[two_frame] two-frame short created successfully")
    return True


def _get_dimensions(video_path: Path) -> tuple[int, int]:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ], capture_output=True, text=True)

    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return 1920, 1080
