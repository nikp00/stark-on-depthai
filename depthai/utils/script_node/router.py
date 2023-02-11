import struct

NAME = "router"

cache = {"mask_z": None, "pos_z": None, "feat_z": None, "state": None}


## Common ##


while True:
    img = node.io["in_img"].get()
    mask = node.io["in_mask"].get()

    new_bbox = node.io["in_new_bbox"].tryGet()
    if new_bbox is None:
        bbox = node.io["in_bbox"].get()

    nn_result = node.io["in_nn"].tryGet()

    if new_bbox is not None:
        cache["state"] = new_bbox
        cache["mask_z"] = None
        cache["pos_z"] = None
        cache["feat_z"] = None

        # node.io["out_img_z"].send(img)
        node.io["out_mask_z"].send(mask)

        log("init")

    if nn_result is not None:
        feat_z = nn_result.getLayerFp16("feat")
        pos_z = nn_result.getLayerFp16("pos")
        mask_z = nn_result.getLayerInt32("mask")

        log(f"feat_z: {feat_z[:5]}")
        log(f"pos_t: {pos_z[:5]}")
        log(f"mask_t: {mask_z[:5]}")

        feat_z = struct.pack(f"<{len(feat_z)}f", *feat_z)
        pos_z = struct.pack(f"<{len(pos_z)}f", *pos_z)
        mask_z = struct.pack(f"<{len(mask_z)}H", *mask_z)

        feat_z_buff = Buffer(len(feat_z))
        pos_z_buff = Buffer(len(pos_z))
        mask_z_buff = Buffer(len(mask_z))

        feat_z_buff.setData(feat_z)
        pos_z_buff.setData(pos_z)
        mask_z_buff.setData(mask_z)

        cache["feat_z"] = feat_z_buff
        cache["pos_z"] = pos_z_buff
        cache["mask_z"] = mask_z_buff

    if cache["mask_z"] is not None:
        log(f"tracking: {nn_result}")

        # node.io["out_img_x"].send(img)
        node.io["out_mask_x"].send(mask)

        # node.io["out_mask_z"].send(cache["mask_z"])
        # node.io["out_pos_z"].send(cache["pos_z"])
        # node.io["out_feat_z"].send(cache["feat_z"])
        if bbox:
            node.io["out_state"].send(bbox)
        else:
            node.io["out_state"].send(cache["state"])
