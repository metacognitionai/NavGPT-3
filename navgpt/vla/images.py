"""Image decoding for the VLA service, derived from VLNCE-EVAL by EPIC Lab (MIT;
see navgpt/vla/LICENSE)."""
import numpy as np
import base64
import cv2


def decode_base64_to_image(base64_string):
    """Decode a base64 string to a numpy RGB image array."""
    image_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
