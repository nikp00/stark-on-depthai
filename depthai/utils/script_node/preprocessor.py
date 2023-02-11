import math
import struct

NAME = "preprocessor"

## Common ##


def calc_bbox(x, y, w, h, crop_sz):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

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

    log(f"crop_sz: {crop_sz}")
    log(f"center_x: {center_x}, center_y: {center_y}, width: {width}, height: {height}")
    log(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
    log(f"x1_pad: {x1_pad}, x2_pad: {x2_pad}, y1_pad: {y1_pad}, y2_pad: {y2_pad}")

    return center_x, center_y, width, height, x1_pad, x2_pad, y1_pad, y2_pad


def make_mask(crop, padding, height, width):
    mask = [0] * height * width

    x, y, w, h = crop
    left, right, top, bottom = padding

    resize_factor_w = width / (w + left + right)
    resize_factor_h = height / (h + top + bottom)

    # Top
    n_top = int(resize_factor_h * top)
    if n_top > 0:
        mask[0 : n_top * width] = [1] * n_top * width

    # Bottom
    n_bottom = int(resize_factor_h * bottom)
    if n_bottom > 0:
        mask[-n_bottom * width :] = [1] * n_bottom * width

    # Left and right
    n_left = int(resize_factor_w * left)
    n_right = int(resize_factor_w * right)
    if n_left > 0 or n_right > 0:
        for i in range(n_top, height - n_bottom):
            if n_left > 0:
                mask[i * width : i * width + n_left] = [1] * n_left
            if n_right > 0:
                mask[i * width + width - n_right : i * width + width] = [1] * n_right

    return mask


def scale_bbox(bbox, scale_h, scale_w):
    x, y, w, h = bbox
    x = int(x * scale_w)
    y = int(y * scale_h)
    w = int(w * scale_w)
    h = int(h * scale_h)
    return x, y, w, h


cache = {"bbox": None}

while True:
    img = node.io["in_img"].get()
    bbox = node.io["in_bbox"].tryGet()
    new_bbox = node.io["in_new_bbox"].tryGet()

    if new_bbox:
        W, H = img.getWidth(), img.getHeight()
        # x, y, w, h = scale_bbox(buffer_to_list(new_bbox.getData()), 480 / H, 640 / W)
        x, y, w, h = buffer_to_list(new_bbox.getData(), dtype=float)

        crop_sz = math.ceil(math.sqrt(w * h) * TEMPLATE_FACTOR)
        resize_factor_value = TEMPLATE_SIZE / crop_sz
        resize_size = TEMPLATE_SIZE
        cache["bbox"] = new_bbox

        log(f"new bbox {[x, y, w, h]}")

    elif bbox:
        W, H = img.getWidth(), img.getHeight()
        # x, y, w, h = scale_bbox(buffer_to_list(new_bbox.getData()), 480 / H, 640 / W)
        x, y, w, h = buffer_to_list(bbox.getData(), dtype=float)
        log(f"bbox {[x, y, w, h]}, search factor: {SEARCH_FACTOR}")
        crop_sz = math.ceil(math.sqrt(w * h) * SEARCH_FACTOR)
        resize_factor_value = SEARCH_SIZE / crop_sz
        resize_size = SEARCH_SIZE

        log(f"bbox {[x, y, w, h]}")

    else:
        bbox = cache["bbox"]
        W, H = img.getWidth(), img.getHeight()
        x, y, w, h = buffer_to_list(bbox.getData(), dtype=float)
        crop_sz = math.ceil(math.sqrt(w * h) * SEARCH_FACTOR)
        resize_factor_value = SEARCH_SIZE / crop_sz
        resize_size = SEARCH_SIZE

        log(f"cached bbox {[x, y, w, h]}")

    if new_bbox or bbox:
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

        mask = make_mask(
            [x, y, w, h],
            [pad_left, pad_right, pad_top, pad_bottom],
            resize_size,
            resize_size,
        )

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

        resize_factor_data = struct.pack("<1f", resize_factor_value)
        resize_factor = Buffer(len(resize_factor_data))
        resize_factor.setData(resize_factor_data)

        mask_buff = Buffer(len(mask))
        mask_buff.setData(bytes(mask))

        node.io["out_cfg_crop"].send(cfg_crop)
        node.io["out_img"].send(img)
        node.io["out_mask"].send(mask_buff)

    if bbox:
        log(f"bbox {[x, y, w, h]}")
        node.io["out_bbox"].send(bbox)
        node.io["out_resize_factor"].send(resize_factor)
