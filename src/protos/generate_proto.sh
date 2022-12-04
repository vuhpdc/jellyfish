#!/bin/bash

: ${root_dir="$HOME/jellyfish/"}
: ${src_path="$root_dir/src/"}

python3 -m grpc_tools.protoc \
	--proto_path=${src_path}/protos/ \
	${src_path}/protos/predict.proto \
	--python_out=${src_path}/protos/ \
	--grpc_python_out=${src_path}/protos/

sed -i '/import predict_pb2 as predict__pb2/c\from . import predict_pb2 as predict__pb2' ${src_path}/protos/predict_pb2_grpc.py
