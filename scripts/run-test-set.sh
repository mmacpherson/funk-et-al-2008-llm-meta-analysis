#!/usr/bin/env bash

set -euo pipefail

script_path="$(realpath "$0")"
script_dir="$(dirname "$script_path")"

yaml_file="${script_dir}/../data/test-annotations.yaml"

# Inline Python script within a heredoc
json_ids=$(python3 <<EOF
import yaml
import json

# Comment to exclude any categories, i.e. for testing/debugging.
# By default runs all categories.
categories = {
    "not-an-empirical-paper",
    "not-an-experiment",
    "traits-not-measured",
    "traits-NOT-integral-to-experimental-design",
    "traits-integral-to-experimental-design",
}

def convert_yaml_to_json_string(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    all_ids = [id for (cat, ids) in data.items() for id in ids if cat in categories]
    return json.dumps(all_ids)

json_string = convert_yaml_to_json_string("$yaml_file")
print(json_string)

EOF
)

set -x
python analyze_papers.py run --dois="${json_ids}" "$@"
