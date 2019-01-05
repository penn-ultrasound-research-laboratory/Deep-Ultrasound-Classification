import json

from src.constants.ultrasoundConstants import (
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL,
    TUMOR_BENIGN,
    TUMOR_MALIGNANT,
    TUMOR_TYPE_LABEL,
    FOCUS_HASH_LABEL,
    FRAME_LABEL,
    SCALE_LABEL)

def frame_image_type_match(frame, image_type):
    """Returns whether a frame type matches the target type. IMAGE_TYPE.ALL always true"""
    if image_type is IMAGE_TYPE.ALL:
        return True
    else:
        return frame[IMAGE_TYPE_LABEL] == image_type.value


def frame_contains_segment(frame):
    """Returns whether a frame has a segment"""
    return FOCUS_HASH_LABEL in frame


def frame_pass_valid_sample_criteria(frame, image_type):
    """Returns whether a frame matches both segment and type criteria"""
    return frame_image_type_match(frame, image_type) and frame_contains_segment(frame)


def get_valid_frame_samples(frames, image_type):
    """Return frames that pass sample criteria. Must match target image type and have segment"""
    return [frame_pass_valid_sample_criteria(frame, image_type) for frame in frames]

def count_valid_frame_samples(frames, image_type):
    return len(get_valid_frame_samples(frames, image_type))


def patient_has_samples(manifest, patient_id, image_type):
    """Returns true if there is at least one frame that passes validity criteria"""
    return len(get_valid_frame_samples(manifest[patient_id], image_type)) > 0


def merge_manifest(path_to_manifest_a, path_to_manifest_b, output_path):
    with open(path_to_manifest_a, 'r') as f:
        manifest_a = json.load(f)

    with open(path_to_manifest_b, 'r') as f:
        manifest_b = json.load(f)

    composite_manifest = { **manifest_a, **manifest_b } 

    with open(output_path, 'w') as f:
        json.dump(composite_manifest, f)
