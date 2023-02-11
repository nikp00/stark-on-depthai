import struct

NAME = "map_bbox"

## Common ##


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


bbox_buffer = Buffer(16)
bbox_data = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

while True:
    output_cords = node.io["in_nn"].get()
    resize_factor = node.io["in_resize_factor"].get()
    state = node.io["in_state"].get()

    output_cords = output_cords.getLayerFp16("outputs_coord")
    resize_factor = buffer_to_list(resize_factor.getData(), float)[0]
    state = buffer_to_list(state.getData(), dtype=float)

    log(f"resize_factor: {resize_factor}, output_cords: {output_cords}")

    pred_box = [0, 0, 0, 0]

    for i in range(4):
        pred_box[i] = output_cords[i] * SEARCH_SIZE / resize_factor

    log(f"pred_box: {output_cords}, state: {state}")

    bbox = clip_box(
        map_box_back(pred_box, state, resize_factor, SEARCH_SIZE),
        IMG_HEIGHT,
        IMG_WIDTH,
        10,
    )

    log(f"bbox: {bbox}")
    bbox = list(map(float, map(int, bbox)))

    bbox_data = struct.pack("<4f", *bbox)
    bbox_buffer.setData(bbox_data)
    node.io["out_bbox"].send(bbox_buffer)
