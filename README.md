# My Personal Scripts Repo

I created this to save time wiht my workflow and to sync my workflow across all my computers.

## .config file
There is an example .config file named .config-example.
Rename any config file to .config if you want to use it. There are a few config options supported:
 - ```default_project_path```: set this equal to the root directory of your projects. This allows for fast fuzzy finding for all your different projects.

### tmux-dev.sh

This opens a fuzzy finder starting from the ~/Programming folder, this will be moved to a .config file soon.
It then opens a tmux session with nvim on the left and 2 panels on the right. The bottom right panel I use for running cmake and other build commands while the top right is typically used for everything else, from git to man pages.




