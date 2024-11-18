#!/bin/bash

# Full path to the Python script
SCRIPT_PATH="/home/user/projects/mon_script.py"

# Check if the Python script is running
if ! pgrep -f "$SCRIPT_PATH" > /dev/null; then
    echo "$SCRIPT_PATH has stopped unexpectedly!" | mail -s "Script Alert" saurabhkumar933@gmail.com
fi

