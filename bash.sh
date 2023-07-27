#!/bin/bash

set -x

echo ~
pwd

# shellcheck disable=SC1090
source ~/.bashrc
source /etc/profile
source /etc/environment

# source /opt/conda/bin/mamba activate bs5
# cat /home/ubuntu/.bashrc

mamba info -e

# mamba run -n bs5 python test.py "$1" "$2" "$3"

# Without following two lines it was throwing error
# /opt/conda/bin/mamba init

# source ~/.bashrc

# /opt/conda/bin/mamba activate bs5

# source /home/ubuntu/.bashrc

mamba run -n bs5 python main.py \
  --api_host "$1" \
  --api_record_id "$2" \
  --api_access_token "$3"

# /opt/conda/bin/python test.py "$1" "$2" "$3"

# shutdown
# sudo shutdown now
