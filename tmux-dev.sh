#!/bin/sh

dir=`cd ~/Programming && find -type d | fzf`
dir="./Programming${dir:1}"

tmux new-session \; send-keys 'cd '$dir" && clear" C-m \; send-keys 'nvim .' C-m \; split-window -h \; send-keys 'cd '$dir" && clear" C-m \; resize-pane -R 25 \; split-window -v \; send-keys 'cd '$dir" && clear" C-m \; resize-pane -D 8 \; select-pane -t 0 \;



