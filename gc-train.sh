TRAINER_PACKAGE_PATH="src"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="grayscale_train_$now"
MAIN_TRAINER_MODULE="src.trainer.task"
STORAGE="research-storage"
JOB_DIR="gs://$STORAGE/$JOB_NAME"
REGION="us-west1"
SCALE_TIER="basic"
DATASET="V2.0_Processed"
MANIFEST="manifest.json"

# ml-engine configs
runtime_version=1.12
python_version=3.5

# Paths to images
IMAGES_PATH="gs://$STORAGE/$DATASET"
MANIFEST_PATH="gs://$STORAGE/$MANIFEST"

# Path to config
# will convert to arg at some point
CONFIG_FILE="default.yaml" 
MODEL_CONFIG_PATH="src/config/$CONFIG_FILE"

# Number workers - will eventually be based on scale-tier
NUM_WORKERS=2

# Local test
# gcloud ml-engine local train --package-path "src" \
#         --module-name $MAIN_TRAINER_MODULE \
#         -- \
#         --images $IMAGES_PATH \
#         --manifest $MANIFEST_PATH \
#         --config $MODEL_CONFIG_PATH \
#         --num-workers $NUM_WORKERS


# Production run
gcloud ml-engine jobs submit training $JOB_NAME \
        --module-name $MAIN_TRAINER_MODULE \
        --package-path $TRAINER_PACKAGE_PATH \
        --scale-tier $SCALE_TIER \
        --job-dir $JOB_DIR \
        --region $REGION \
        --runtime-version $runtime_version \
        --python-version $python_version \
        -- \
        --images $IMAGES_PATH \
        --manifest $MANIFEST_PATH \
        --config $MODEL_CONFIG_PATH \
        --num-workers $NUM_WORKERS
