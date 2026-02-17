#!/usr/bin/env bash
set -euo pipefail

# Defaults (change as you like) 
# export JOB_NAME=beidchen-$1-$(date +%H%M) 
export JOB_NAME="beidchen-$1-$(date +%H%M%S)"
export NUM_NODES=$2 
export SCRIPT_NAME=$3 

# envsubst '${JOB_NAME} ${NUM_NODES} ${SCRIPT_NAME}' < templatenodeswithinterneth200.tmpl.yaml > scripts/${JOB_NAME}.yaml 
envsubst '${JOB_NAME} ${NUM_NODES} ${SCRIPT_NAME}' < template.yaml > scripts/${JOB_NAME}.yaml 

echo "Script ${JOB_NAME}.yaml created"