#!/bin/sh

script_path="$(dirname "$BASH_SOURCE")"

dir=$(eval cd "$script_path" && find -type f \( -name "*.sh" -and ! -name "script-launcher.sh" \) | fzf)
eval dir="${dir}"

cd $script_path
. $dir


