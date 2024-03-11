export DEBUG_STEPS=1000
ZERO_STAGE=0 COMPILE=false bash ./run.sh
ZERO_STAGE=0 COMPILE=true bash ./run.sh
ZERO_STAGE=1 COMPILE=false bash ./run.sh
ZERO_STAGE=1 COMPILE=true bash ./run.sh
ZERO_STAGE=2 COMPILE=false bash ./run.sh
ZERO_STAGE=2 COMPILE=true bash ./run.sh
ZERO_STAGE=3 COMPILE=false bash ./run.sh
ZERO_STAGE=3 COMPILE=true bash ./run.sh
