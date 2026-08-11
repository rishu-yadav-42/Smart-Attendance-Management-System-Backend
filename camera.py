import threading
import time
import cv2

try:
    from backend.config import CAMERA_FRAME_RETRY_LIMIT
except ImportError:
    from config import CAMERA_FRAME_RETRY_LIMIT

camera_stream_should_stop = False
_frame_buffer = []
_frame_buffer_lock = threading.Lock()
_MAX_BUFFER_SIZE = 50


def store_frame(frame):
    global _frame_buffer
    with _frame_buffer_lock:
        _frame_buffer.append(frame.copy())
        if len(_frame_buffer) > _MAX_BUFFER_SIZE:
            _frame_buffer.pop(0)


def get_buffered_frames():
    with _frame_buffer_lock:
        return list(_frame_buffer)


def clear_frame_buffer():
    global _frame_buffer
    with _frame_buffer_lock:
        _frame_buffer.clear()


def open_camera(camera_index=0):
    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("DEFAULT", None),
    ]

    for name, backend in backends:
        cam = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
        if not cam or not cam.isOpened():
            if cam:
                cam.release()
            continue

        try:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        stable_frames = 0
        for _ in range(CAMERA_FRAME_RETRY_LIMIT):
            try:
                ret, frame = cam.read()
                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                    stable_frames += 1
                    if stable_frames >= 3:
                        break
                else:
                    stable_frames = 0
            except Exception:
                stable_frames = 0
            time.sleep(0.1)

        if stable_frames >= 3:
            time.sleep(0.2)
            return cam

        cam.release()
        time.sleep(0.2)

    return None


def stop_camera_stream():
    global camera_stream_should_stop
    camera_stream_should_stop = True


def reset_camera_stream():
    global camera_stream_should_stop
    camera_stream_should_stop = False


def gen_frames(max_duration=15):
    reset_camera_stream()
    clear_frame_buffer()
    cam = open_camera(0)
    if cam is None:
        return

    start_time = time.time()
    try:
        while True:
            if camera_stream_should_stop:
                break
            if time.time() - start_time > max_duration:
                break
            success, frame = cam.read()
            if not success:
                time.sleep(0.05)
                continue

            store_frame(frame)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)
    finally:
        cam.release()
