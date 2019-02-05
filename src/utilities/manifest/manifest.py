import json
import os

from src.constants.ultrasound import (
    FRAME_LABEL,
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

def frame_pass_valid_sample_criteria(frame, image_type):
    """Returns whether a frame matches type criteria"""
    return frame_image_type_match(frame, image_type)


def get_valid_frame_samples(frames, image_type):
    """Return frames that pass sample criteria. Must match target image type"""
    return [f for f in frames if frame_pass_valid_sample_criteria(f, image_type)]

def count_valid_frame_samples(frames, image_type):
    return len(get_valid_frame_samples(frames, image_type))


def patient_has_samples(manifest, patient_id, image_type):
    """Returns true if there is at least one frame that passes validity criteria"""
    return count_valid_frame_samples(manifest[patient_id], image_type) > 0
    

def merge_manifest(path_to_manifest_a, path_to_manifest_b, output_path):
    with open(path_to_manifest_a, 'r') as f:
        manifest_a = json.load(f)

    with open(path_to_manifest_b, 'r') as f:
        manifest_b = json.load(f)

    composite_manifest = { **manifest_a, **manifest_b } 

    with open(output_path, 'w') as f:
        json.dump(composite_manifest, f)
        

def convert_old_manifest_to_new_format(path_to_manifest, path_to_images):
    with open(path_to_manifest, 'r') as f:
        manifest = json.load(f)

    for patient_id in manifest:
        remove_frames = []
        for frame in manifest[patient_id]:
            # Always get rid of the "FOCUS" key
            frame.pop(FOCUS_HASH_LABEL, None)
            
            type_path = "Benign" if frame[TUMOR_TYPE_LABEL] == TUMOR_BENIGN else "Malignant"
            # If the frame does not exist, remove the frame
            if not os.path.isfile("{0}/{1}/{2}/{3}".format(path_to_images, type_path, patient_id, frame[FRAME_LABEL])):

                # Optional logging
                # print("Missing: {0}/{1}".format(patient_id, frame[FRAME_LABEL]))
                
                remove_frames.append(frame[FRAME_LABEL])

        # Optional logging
        # print("{0}: {1}".format(patient_id, remove_frames))

        manifest[patient_id] = [frame for frame in manifest[patient_id] if frame[FRAME_LABEL] not in remove_frames]
    
    with open(path_to_manifest, 'w') as f:
        json.dump(manifest, f)
