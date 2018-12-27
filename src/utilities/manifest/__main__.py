import sys

from src.utilities.manifest.manifestUtilities import merge_manifest

merge_manifest(sys.argv[1], sys.argv[2], sys.argv[3])