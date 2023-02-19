import struct
import math
import time

NAME = "manager"
TEMPLATE_SIZE = 128
TEMPLATE_FACTOR = 2.0
SEARCH_SIZE = 320
SEARCH_FACTOR = 5.0
IMG_WIDTH = 640
IMG_HEIGHT = 480
LOG_LEVEL = -1


class BufferMgr:
    def __init__(self):
        self._bufs = {}

    def __call__(self, size):
        try:
            buf = self._bufs[size]
        except KeyError:
            buf = self._bufs[size] = Buffer(size)
        return buf


def log(msg, level=0):
    if level <= LOG_LEVEL:
        level_label = ["DEBUG", "INFO"][level - 1]
        header = f"[{NAME}][{level_label}]"
        node.warn(f"{header:<20}    {msg}")


def scale_bbox(bbox, scale_h, scale_w):
    x, y, w, h = bbox
    x = int(x * scale_w)
    y = int(y * scale_h)
    w = int(w * scale_w)
    h = int(h * scale_h)
    return x, y, w, h


def buffer_to_list(buffer, dtype=int):
    if dtype == int:
        return struct.unpack("<" + "H" * (len(buffer) // 2), buffer)
    elif dtype == float:
        return struct.unpack("<" + "f" * (len(buffer) // 4), buffer)


def calc_bbox(x, y, w, h, crop_sz):
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - IMG_WIDTH + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - IMG_HEIGHT + 1, 0)

    center_x = int(x1 + (x2 - x1) // 2)
    center_y = int(y1 + (y2 - y1) // 2)
    height = int(y1_pad + y2_pad + (y2 - y1))
    width = int(x1_pad + x2_pad + (x2 - x1))

    log(f"crop_sz: {crop_sz}", 1)
    log(
        f"center_x: {center_x}, center_y: {center_y}, width: {width}, height: {height}",
        1,
    )
    log(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}", 1)
    log(f"x1_pad: {x1_pad}, x2_pad: {x2_pad}, y1_pad: {y1_pad}, y2_pad: {y2_pad}", 1)

    return center_x, center_y, width, height, x1_pad, x2_pad, y1_pad, y2_pad


def clip_box(box, H, W, margin):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]


def map_box_back(pred_box, state, resize_factor, search_size):
    cx_prev, cy_prev = (
        state[0] + 0.5 * state[2],
        state[1] + 0.5 * state[3],
    )
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def get_ms():
    return time.monotonic() * 1000


buff_mgr = BufferMgr()
state = None
fet_z = None
mask_z = None
pos_z = None
backbone_result = None
complete_result = None
timer_start = 0

initialized = False


log(f"Starting main loop  |  LOG LEVEL: {LOG_LEVEL}", 0)
while True:
    timer_start = get_ms()

    img = node.io["in_resized_img"].get()
    original_image = node.io["in_img"].get()
    new_bbox = node.io["in_new_bbox"].tryGet()

    log(f"Received data  | {get_ms() - timer_start} ms", 0)
    timer_start = get_ms()

    if new_bbox:
        initialized = True
        state = buffer_to_list(new_bbox.getData(), dtype=float)

        log(f"Received new bbox {state}  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

        x, y, w, h = scale_bbox(
            state,
            IMG_HEIGHT / original_image.getHeight(),
            IMG_WIDTH / original_image.getWidth(),
        )

        crop_sz = math.ceil(math.sqrt(w * h) * TEMPLATE_FACTOR)
        resize_factor = TEMPLATE_SIZE / crop_sz
        resize_size = TEMPLATE_SIZE

        (
            center_x,
            center_y,
            width,
            height,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        ) = calc_bbox(x, y, w, h, crop_sz)

        rotated_rect = RotatedRect()
        rotated_rect.center.x = center_x
        rotated_rect.center.y = center_y
        rotated_rect.size.width = width
        rotated_rect.size.height = height
        rotated_rect.angle = 0

        cfg_crop = ImageManipConfig()
        cfg_crop.setCropRotatedRect(rotated_rect, False)
        cfg_crop.setResize(resize_size, resize_size)
        cfg_crop.setKeepAspectRatio(True)
        cfg_crop.setFrameType(ImgFrame.Type.BGR888p)

        log(f"Sending to backbone...  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

        node.io["out_img_backbone"].send(img)
        node.io["out_cfg_crop_backbone"].send(cfg_crop)

        # Wait for backbone to finish
        log(f"Waiting for backbone...  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

        backbone_result = node.io["in_backbone_result"].get()

        log(f"Backbone finished...  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

    elif initialized:
        x, y, w, h = state
        crop_sz = math.ceil(math.sqrt(w * h) * SEARCH_FACTOR)
        resize_factor = SEARCH_SIZE / crop_sz
        resize_size = SEARCH_SIZE

        (
            center_x,
            center_y,
            width,
            height,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        ) = calc_bbox(x, y, w, h, crop_sz)

        rotated_rect = RotatedRect()
        rotated_rect.center.x = center_x
        rotated_rect.center.y = center_y
        rotated_rect.size.width = width
        rotated_rect.size.height = height
        rotated_rect.angle = 0

        cfg_crop = ImageManipConfig()
        cfg_crop.setCropRotatedRect(rotated_rect, False)
        cfg_crop.setResize(resize_size, resize_size)
        cfg_crop.setKeepAspectRatio(True)
        cfg_crop.setFrameType(ImgFrame.Type.BGR888p)

        log(f"Sending to complete...  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

        node.io["out_img_complete"].send(img)
        node.io["out_cfg_crop_complete"].send(cfg_crop)
        node.io["out_backbone_result"].send(backbone_result)

        # Wait for complete to finish
        log(f"Waiting for complete...  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

        complete_result = node.io["in_complete_result"].get()

        log(f"Complete finished...  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

        output_cord = complete_result.getLayerFp16("outputs_coord")

        log(f"Output cord: {output_cord}  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

        pred_box = [0, 0, 0, 0]
        for i in range(4):
            pred_box[i] = output_cord[i] * SEARCH_SIZE / resize_factor

        state = clip_box(
            map_box_back(pred_box, state, resize_factor, SEARCH_SIZE),
            IMG_HEIGHT,
            IMG_WIDTH,
            20,
        )

        log(f"New state: {state}  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()

        scaled_state = scale_bbox(
            state,
            original_image.getHeight() / IMG_HEIGHT,
            original_image.getWidth() / IMG_WIDTH,
        )

        bbox_data = struct.pack("<4f", *scaled_state)
        bbox_buff = buff_mgr(len(bbox_data))
        bbox_buff.getData()[:] = bbox_data
        node.io["out_bbox"].send(bbox_buff)

        log(f"Sent bbox  |  {get_ms() - timer_start} ms", 0)
        timer_start = get_ms()
