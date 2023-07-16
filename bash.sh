#!/bin/bash

# shellcheck disable=SC1090
source /home/ubuntu/.bashrc

mamba activate bs5

python main.py \
  --api_host "$1" \
  --api_record_id "$2" \
  --api_access_token "$3"

# shutdown
sudo shutdown now
