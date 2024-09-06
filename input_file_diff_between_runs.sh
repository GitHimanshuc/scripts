#!/bin/bash

# This script compares .input files between two directories specified as command-line arguments.
# It outputs the differences in files with the same name in both directories.

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_first_directory> <path_to_second_directory>"
    exit 1
fi

# Assign command-line arguments to variables
DIR1="$1"
DIR2="$2"

# Function to compare files
compare_files() {
    local dir1="$1"
    local dir2="$2"

    # Loop through all .input files in the first directory
    for file in "$dir1"/*.input; do
        # Extract the file name from the full path
        local filename=$(basename "$file")

        # Check if the same file exists in the second directory
        if [ -f "$dir2/$filename" ]; then
            echo "Comparing $filename:"
            diff "$file" "$dir2/$filename"
            echo "----------------------------------------"
        else
            echo "$filename exists in $dir1 but not in $dir2"
        fi
    done
}

# Verify that both directories exist
if [ ! -d "$DIR1" ]; then
    echo "Directory $DIR1 does not exist. Please check the path and try again."
    exit 1
fi

if [ ! -d "$DIR2" ]; then
    echo "Directory $DIR2 does not exist. Please check the path and try again."
    exit 1
fi

# Call the function to compare files
compare_files "$DIR1" "$DIR2"