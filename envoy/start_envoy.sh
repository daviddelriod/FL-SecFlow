#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --envoy-config-path envoy_config.yaml -dh ip-172-31-0-197.ec2.internal -dp 4444