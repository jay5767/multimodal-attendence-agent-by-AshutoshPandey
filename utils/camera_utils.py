"""Camera utility module for AURORA.

Detects available webcam and video capture devices on Windows and Linux,
probes compatible DirectShow/MediaFoundation APIs with caching and safe handle release.
"""

import cv2
from typing import List, Dict, Any, Optional
from utils.logging_config import logger

_CACHED_CAMERAS: Optional[List[Dict[str, Any]]] = None


def get_available_cameras(max_devices_to_scan: int = 4, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Scan device indices to find functional video capture devices with caching.
    
    Returns:
        List of dicts with device 'index', 'backend_id', 'backend_name', 'name', 'resolution'
    """
    global _CACHED_CAMERAS
    if _CACHED_CAMERAS is not None and not force_refresh:
        return _CACHED_CAMERAS

    found_devices = []
    
    # Try DirectShow first, then default
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    for idx in range(max_devices_to_scan):
        for backend_id, backend_name in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, backend_id)
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        h, w = frame.shape[:2]
                        found_devices.append({
                            "index": idx,
                            "backend_id": backend_id,
                            "backend_name": backend_name,
                            "name": f"Camera Device #{idx} ({backend_name} • {w}x{h})",
                            "resolution": (w, h)
                        })
                        cap.release()
                        cap = None
                        break
            except Exception as e:
                logger.debug(f"Device scan exception for idx {idx} with {backend_name}: {e}")
            finally:
                if cap is not None:
                    try:
                        if cap.isOpened():
                            cap.release()
                    except Exception:
                        pass
                    
    _CACHED_CAMERAS = found_devices
    return found_devices


def get_default_camera_index() -> int:
    """Get the primary working camera index, defaulting to 0 if none detected."""
    cams = get_available_cameras()
    if cams:
        logger.info(f"Auto-detected working camera at index {cams[0]['index']} ({cams[0]['backend_name']})")
        return cams[0]["index"]
    return 0
