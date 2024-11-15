# Get the name of the current directory
base_dir=$(basename "$PWD")

# Step 1: Delete all non-.mp4 files in the current directory and subdirectories
find "$base_dir" -type f ! -name "*.mp4" -exec rm -f {} +

# Step 2: Delete all empty directories in the current directory
find "$base_dir" -type d -empty -delete

# Step 3: Rename directories with "@" in their names by removing everything after "@"
for dir in "$base_dir"/*@*; do
    mv "$dir" "${dir%@*}"
done

# Step 4: Rename .mp4 files in "save" subdirectories to match the name of their grandparent directory
for file in "$base_dir"/*/save/*.mp4; do
    dir_name=$(basename "$(dirname "$(dirname "$file")")")
    mv "$file" "$(dirname "$file")/${dir_name}.mp4"
done

# Step 5: Create the "post-processing" directory if it doesn't already exist
mkdir -p post-processing

# Step 6: Copy all .mp4 files from the base directory into the "post-processing" directory
find "$base_dir" -name "*.mp4" -exec cp {} post-processing/ \;
