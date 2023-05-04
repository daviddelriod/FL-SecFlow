#!/bin/bash
set -e

fx envoy start -n env_two --disable-tls --envoy-config-path envoy_config2.yaml -dh 52.20.29.134 -dp 4444