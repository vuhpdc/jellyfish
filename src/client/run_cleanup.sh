#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: ./run_cleanup.sh <network_interface>"
	exit
fi

network_iface=$1
sudo tc qdisc del dev ${network_iface} root
