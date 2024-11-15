#!/bin/bash

mkdir -p Samples

find . -type f -name "*.mp4" | while read -r filepath; do
    dirpath=$(dirname "$filepath")
    filename=$(basename "$filepath" .mp4)
    gifname="${dirpath//\//_}_${filename}.gif"
    ffmpeg -i "$filepath" -vf "fps=10,scale=320:-1" -loop 0 "Samples/$gifname"
    echo "Converted $filepath to Samples/$gifname"
done
