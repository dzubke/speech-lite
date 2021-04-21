#!/bin/bash

echo $( date )
threshold=0.15
n_cpu=$( grep 'model name' /proc/cpuinfo | wc -l )
threshold=$( echo $n_cpu*$threshold | bc )
echo threshold: $threshold

count=0
while true
do

  #idle=$( mpstat | awk 'FNR == 4' | awk '{print $13}' )
  #if (( $(echo $idle'>'$idle_threshold | bc -l) ))
  load=$(uptime | sed -e 's/.*load average: //g' | awk '{ print $3 }')
  if (( $(echo $load'<'$threshold | bc -l) ))
  then
    echo "Idling.."
    ((count+=1))
  fi
  echo "Idle minutes count = $count"

  if (( count>10 ))
  then
    echo Shutting down
    # wait a little bit more before actually pulling the plug
    sleep 300
    sudo poweroff
  fi

  sleep 60

done
