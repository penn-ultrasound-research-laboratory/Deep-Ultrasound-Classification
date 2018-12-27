import json

def merge_manifest(path_to_manifest_a, path_to_manifest_b, output_path):
    with open(path_to_manifest_a, 'r') as f:
        manifest_a = json.load(f)

    with open(path_to_manifest_b, 'r') as f:
        manifest_b = json.load(f)

    composite_manifest = { **manifest_a, **manifest_b } 

    with open(output_path, 'w') as f:
        json.dump(composite_manifest, f)
