TRAINER_PACKAGE_PATH="/Users/Matthew/Documents/Research/ultrasound-ml-engine/src"
MAIN_TRAINER_MODULE="trainer.task"
STORAGE_BUCKET="research-storage"

DATASET="V2.0_Processed"
MANIFEST="manifest.json"
CONFIG_FILE="default.yaml" # will convert to arg at some point

REGION="us-west1"
SCALE_TIER="BASIC_GPU"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="grayscale_train_$now"
JOB_DIR="gs://$STORAGE_BUCKET/staging/$JOB_NAME"

# ml-engine configs
runtime_version=1.12
python_version=3.5

# Paths to images
IMAGES_PATH="gs://$STORAGE_BUCKET/$DATASET"
MANIFEST_PATH="gs://$STORAGE_BUCKET/$MANIFEST"

# Path to config
MODEL_CONFIG_PATH="gs://$STORAGE_BUCKET/config/$CONFIG_FILE"

# Number workers - will eventually be based on scale-tier
NUM_WORKERS=2

# Production run
gcloud ml-engine jobs submit training $JOB_NAME \
        --module-name $MAIN_TRAINER_MODULE \
        --package-path $TRAINER_PACKAGE_PATH \
        --job-dir $JOB_DIR \
        --scale-tier $SCALE_TIER \
        --region $REGION \
        --runtime-version $runtime_version \
        --python-version $python_version \
        -- \
        --images $IMAGES_PATH \
        --manifest $MANIFEST_PATH \
        --config $MODEL_CONFIG_PATH \
        --num-workers $NUM_WORKERS