from collections import OrderedDict

def safe_manual_rpn_call(rpn, features, images, image_shapes):
    if isinstance(features, list):
        features = OrderedDict((str(i), f) for i, f in enumerate(features))
    return rpn(features, images, image_shapes)

def safe_box_roi_pool(pooler, features, proposals, image_shapes):
    if isinstance(features, list):
        print("[MANUAL POOL] Fixing features passed as list.")
        features = OrderedDict((str(i), f) for i, f in enumerate(features))
    return pooler(features, proposals, image_shapes)

def to_dict(features):
    if isinstance(features, OrderedDict):
        return features
    elif isinstance(features, dict):
        return OrderedDict(features)
    elif isinstance(features, list):
        return OrderedDict((str(i), f) for i, f in enumerate(features))
    else:
        raise TypeError(f"[DEBUG to_dict] Unexpected feature type: {type(features)}")