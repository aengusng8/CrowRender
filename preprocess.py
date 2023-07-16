import torch


def encode_from_custom_annotation(custom_annotations, size=512):
    #     custom_annotations = [
    #     {'x': 83, 'y': 335, 'width': 70, 'height': 69, 'label': 'blue metal cube'},
    #     {'x': 162, 'y': 302, 'width': 110, 'height': 138, 'label': 'blue metal cube'},
    #     {'x': 274, 'y': 250, 'width': 191, 'height': 234, 'label': 'blue metal cube'},
    #     {'x': 14, 'y': 18, 'width': 155, 'height': 205, 'label': 'blue metal cube'},
    #     {'x': 175, 'y': 79, 'width': 106, 'height': 119, 'label': 'blue metal cube'},
    #     {'x': 288, 'y': 111, 'width': 69, 'height': 63, 'label': 'blue metal cube'}
    # ]
    H, W = size, size

    objects = []
    for j in range(len(custom_annotations)):
        xyxy = [
            custom_annotations[j]["x"],
            custom_annotations[j]["y"],
            custom_annotations[j]["x"] + custom_annotations[j]["width"],
            custom_annotations[j]["y"] + custom_annotations[j]["height"],
        ]
        objects.append(
            {
                "caption": custom_annotations[j]["label"],
                "bbox": xyxy,
            }
        )

    out = encode_scene(
        objects, H=H, W=W, src_bbox_format="xyxy", tgt_bbox_format="xyxy"
    )

    return out


def encode_scene(
    obj_list, H=320, W=320, src_bbox_format="xywh", tgt_bbox_format="xyxy"
):
    """Encode scene into text and bounding boxes
    Args:
        obj_list: list of dicts
            Each dict has keys:

                'color': str
                'material': str
                'shape': str
                or
                'caption': str

                and

                'bbox': list of 4 floats (unnormalized)
                    [x0, y0, x1, y1] or [x0, y0, w, h]
    """
    box_captions = []
    for obj in obj_list:
        if "caption" in obj:
            box_caption = obj["caption"]
        else:
            box_caption = f"{obj['color']} {obj['material']} {obj['shape']}"
        box_captions += [box_caption]

    assert src_bbox_format in [
        "xywh",
        "xyxy",
    ], f"src_bbox_format must be 'xywh' or 'xyxy', not {src_bbox_format}"
    assert tgt_bbox_format in [
        "xywh",
        "xyxy",
    ], f"tgt_bbox_format must be 'xywh' or 'xyxy', not {tgt_bbox_format}"

    boxes_unnormalized = []
    boxes_normalized = []
    for obj in obj_list:
        if src_bbox_format == "xywh":
            x0, y0, w, h = obj["bbox"]
            x1 = x0 + w
            y1 = y0 + h
        elif src_bbox_format == "xyxy":
            x0, y0, x1, y1 = obj["bbox"]
            w = x1 - x0
            h = y1 - y0
        assert x1 > x0, f"x1={x1} <= x0={x0}"
        assert y1 > y0, f"y1={y1} <= y0={y0}"
        assert x1 <= W, f"x1={x1} > W={W}"
        assert y1 <= H, f"y1={y1} > H={H}"

        if tgt_bbox_format == "xywh":
            bbox_unnormalized = [x0, y0, w, h]
            bbox_normalized = [x0 / W, y0 / H, w / W, h / H]

        elif tgt_bbox_format == "xyxy":
            bbox_unnormalized = [x0, y0, x1, y1]
            bbox_normalized = [x0 / W, y0 / H, x1 / W, y1 / H]

        boxes_unnormalized += [bbox_unnormalized]
        boxes_normalized += [bbox_normalized]

    assert len(box_captions) == len(
        boxes_normalized
    ), f"len(box_captions)={len(box_captions)} != len(boxes_normalized)={len(boxes_normalized)}"

    text = prepare_text(box_captions, boxes_normalized)

    out = {}
    out["text"] = text
    out["box_captions"] = box_captions
    out["boxes_normalized"] = boxes_normalized
    out["boxes_unnormalized"] = boxes_unnormalized

    return out


def prepare_text(
    box_captions=[],
    box_normalized=[],
    global_caption=None,
    # image_resolution=512,
    text_reco=True,
    num_bins=1000,
    # tokenizer=None
    spatial_text=False,
):
    # Describe box shape in text
    if spatial_text:
        # box_descriptions = []
        # for box_sample_ii in range(len(box_captions)):
        #     box = box_normalized[box_sample_ii]
        #     box_caption = box_captions[box_sample_ii]
        #     box_description = prepare_spatial_description(box, box_caption)
        #     box_descriptions.append(box_description)
        # text = " ".join(box_descriptions)
        raise NotImplementedError

    # Describe box
    else:
        box_captions_with_coords = []

        if isinstance(box_normalized, torch.Tensor):
            box_normalized = box_normalized.tolist()

        for box_sample_ii in range(len(box_captions)):
            box = box_normalized[box_sample_ii]
            box_caption = box_captions[box_sample_ii]

            # print(box_caption)

            # quantize into bins
            quant_x0 = int(round((box[0] * (num_bins - 1))))
            quant_y0 = int(round((box[1] * (num_bins - 1))))
            quant_x1 = int(round((box[2] * (num_bins - 1))))
            quant_y1 = int(round((box[3] * (num_bins - 1))))

            if text_reco:
                # ReCo format
                # Add SOS/EOS before/after regional caption
                SOS_token = "<|startoftext|>"
                EOS_token = "<|endoftext|>"
                box_captions_with_coords += [
                    f"<bin{str(quant_x0).zfill(3)}>",
                    f"<bin{str(quant_y0).zfill(3)}>",
                    f"<bin{str(quant_x1).zfill(3)}>",
                    f"<bin{str(quant_y1).zfill(3)}>",
                    SOS_token,
                    box_caption,
                    EOS_token,
                ]

            else:
                box_captions_with_coords += [
                    f"<bin{str(quant_x0).zfill(3)}>",
                    f"<bin{str(quant_y0).zfill(3)}>",
                    f"<bin{str(quant_x1).zfill(3)}>",
                    f"<bin{str(quant_y1).zfill(3)}>",
                    box_caption,
                ]

        text = " ".join(box_captions_with_coords)

    if global_caption is not None:
        # Global caption
        if text_reco:
            # ReCo format
            # Add SOS/EOS before/after regional caption
            # SOS_token = '<|startoftext|>'
            EOS_token = "<|endoftext|>"
            # global_caption = f"{SOS_token} {global_caption} {EOS_token}"

            # SOS token will be automatically added
            global_caption = f"{global_caption} {EOS_token}"

        text = f"{global_caption} {text}"

    return text
