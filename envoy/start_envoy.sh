#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --envoy-config-path envoy_config.yaml -dh 172.31.0.197 -dp 50051