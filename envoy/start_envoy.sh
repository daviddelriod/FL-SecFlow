#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --envoy-config-path envoy_config.yaml -dh 52.20.29.134	 -dp 50051