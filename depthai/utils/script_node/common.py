import struct

TEMPLATE_SIZE = 128
TEMPLATE_FACTOR = 2.0
SEARCH_SIZE = 320
SEARCH_FACTOR = 5.0
IMG_WIDTH = 640
IMG_HEIGHT = 480


def log(msg):
    header = f"[{NAME}]"
    node.warn(f"{header:<15}    {msg}")


def buffer_to_list(buffer, dtype=int):
    if dtype == int:
        return struct.unpack("<" + "H" * (len(buffer) // 2), buffer)
    elif dtype == float:
        return struct.unpack("<" + "f" * (len(buffer) // 4), buffer)
