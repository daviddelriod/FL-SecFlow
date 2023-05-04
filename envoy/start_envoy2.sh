#!/bin/bash
set -e

fx envoy start -n env_two --disable-tls --envoy-config-path envoy_config2.yaml -dh ec2-52-20-29-134.compute-1.amazonaws.com -dp 4444