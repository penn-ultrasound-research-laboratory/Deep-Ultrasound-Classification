TRAINER_PACKAGE_PATH="/Users/Matthew/Documents/Research/ultrasound-ai-platform/src"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="grayscale_train_$now"
MAIN_TRAINER_MODULE="trainer.task"
STORAGE="research-storage"
JOB_DIR="gs://$STORAGE/$JOB_NAME"
PACKAGE_STAGING_PATH="gs://$STORAGE/staging"
REGION="us-west1"
SCALE_TIER="basic"
DATASET="V2.0_Processed"
MANIFEST="manifest.json"

# ai-platform configs
runtime_version=1.12
python_version=3.5

# Paths to images
IMAGES_PATH="gs://$STORAGE/$DATASET"
MANIFEST_PATH="gs://$STORAGE/$MANIFEST"

# Path to config
CONFIG_FILE="local.yaml" 
MODEL_CONFIG_PATH="src/config/$CONFIG_FILE"

# Number workers - will eventually be based on scale-tier
NUM_WORKERS=2

# Production run
gcloud ai-platform local train --package-path $TRAINER_PACKAGE_PATH \
        --module-name $MAIN_TRAINER_MODULE \
        -- \
        --images $IMAGES_PATH \
        --manifest $MANIFEST_PATH \
        --config $MODEL_CONFIG_PATH \
        --num-workers $NUM_WORKERS