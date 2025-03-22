#!/bin/sh

CONFIG_FILE=.config

default_project_path=""

IFS=$'\n' # this sets the Internel Field Separator to \n
for LINE in $(cat "$CONFIG_FILE")
do
	if [[ $LINE == //* ]];
	then
		continue
	fi

	if [[ $LINE == default_project_path=* ]];
	then
		default_project_path=${LINE:21}
	fi
done
echo "Default path loaded: $default_project_path"

dir=$(eval cd "$default_project_path" && find -type d | fzf)
eval dir="$default_project_path${dir:1}"
echo "Final project path: $dir"


tmux new-session \; send-keys 'cd '$dir" && clear" C-m \; send-keys 'nvim .' C-m \; split-window -h \; send-keys 'cd '$dir" && clear" C-m \; resize-pane -R 25 \; split-window -v \; send-keys 'cd '$dir" && clear" C-m \; resize-pane -D 8 \; select-pane -t 0 \;
 


