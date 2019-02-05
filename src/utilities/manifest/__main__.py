import sys

from src.utilities.manifest.manifest import convert_old_manifest_to_new_format

convert_old_manifest_to_new_format(sys.argv[1], sys.argv[2])