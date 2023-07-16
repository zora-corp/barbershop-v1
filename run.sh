#!/bin/bash

cd Barbershop || exit 1
nohup /bin/bash bash.sh "$1" "$2" "$3" || \
 curl --head -H "Authorization: Bearer $3" "https://$1/api/v3/gan-based-generated-styles/$2/errorAlarm"> my.log 2>&1 &
# echo $! > save_pid.txt
pid=$!
echo "{\"pid\": \"$pid\"}"