#!/bin/bash
set -e

fx envoy start -n env_two --disable-tls --envoy-config-path envoy_config2.yaml -dh director -dp 50051