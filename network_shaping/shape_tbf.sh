#!/bin/bash
export LC_NUMERIC=en_US.UTF-8
# set -x
if [ "$#" -ne 7 ]; then
  echo "Usage: ./shape_tbf.sh <network_interface> <shaping_data_file> <shaping_interval> <src_port> <dst_ip> <dst_port> <my_rank>"
  exit
fi

network_iface=$1
input_file=$2
shaping_interval=$3
SRC_PORT=$4
SINK=$5
DST_PORT=$6
MY_BAND=$(( $7 % 8 + 3 ))
MY_HANDLE=$(( $MY_BAND + 10 ))

mapfile -t list < $input_file
total_values=$(echo "${#list[@]}")
start_idx=$(( $RANDOM % $total_values ))
bw_values_arr=( "${list[@]:start_idx}" "${list[@]:0:start_idx}" )

tc_adapter=${network_iface} # host iface
tbf_latency=10000
tbf_burst=1540


function init() {
  echo "ts,bw"
  
  tc qdisc add dev $tc_adapter root handle 1: prio bands 16
  
  tc qdisc add dev $tc_adapter parent 1:${MY_BAND} handle ${MY_HANDLE}: tbf rate 100Mbit latency "${tbf_latency}"ms burst ${tbf_burst} 

  # @NOTE: Uncomment the below line for enabling shaping at IP level
  # tc filter add dev $tc_adapter protocol ip parent 1: prio 1 u32 match ip dst ${SINK} flowid 1:${MY_BAND}

  # @NOTE: Uncomment the below line for enabling shaping at source port level
  # tc filter add dev $tc_adapter protocol ip parent 1: prio 1 u32 match ip sport ${SRC_PORT} 0xffff flowid 1:1
  tc filter add dev $tc_adapter protocol ip parent 1: prio 1 u32 match ip sport ${SRC_PORT} 0xffff flowid 1:${MY_BAND}

  # @NOTE: Uncomment the below line for enabling shaping at dst port level
  # tc filter add dev $tc_adapter protocol ip parent 1: prio 1 u32 match ip dport ${DST_PORT} 0xffff flowid 1:1

  # Latency delay
  # tc qdisc add dev $tc_adapter parent 1:${MY_BAND} handle 5: netem delay 10ms
}

function clear() {
  # @NOTE: Do not delete the root qdisc
  # tc qdisc del dev $tc_adapter root

  # Delete the qdisc of this prio band
  tc qdisc del dev $tc_adapter parent 1:${MY_BAND} handle ${MY_HANDLE}:
}

function stop() {
  clear
  exit
}

function change_bw() {
  timestamp=$(date +"%T.%3N")
  echo "$timestamp,$1"
  if [ 1 -eq "$(echo "$1 <= 5" | bc)" ]; then
    tbf_burst=1540
  fi
  tc qdisc change dev $tc_adapter parent 1:${MY_BAND} handle ${MY_HANDLE}: tbf rate "$1"Mbit latency "${tbf_latency}"ms burst ${tbf_burst}
}

trap "stop" SIGHUP SIGINT SIGTERM

clear
init

TIMESTAMP=$(sleepenh 0)
while true
do
  for item in ${bw_values_arr[@]}
  do
          if [ -z "${item//[$'\t\r\n ']}" ] #skip empty var
          then 
                  continue
          fi
          
          # Adjust bandwidth
          change_bw "${item//[$'\t\r\n ']}"
        
          TIMESTAMP=$(sleepenh $TIMESTAMP $shaping_interval)
  done
done

t=$(sleepenh $t 3)

stop
