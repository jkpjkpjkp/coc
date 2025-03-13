#!/usr/bin/env fish

# Check if any directory names are provided as arguments
if test (count $argv) -eq 0
    echo "Usage: $argv[0] dir1 [dir2 ...]"
    exit 1
end

# Define the output Markdown file
set output "output.md"

# Clear the output file if it exists, or create it if it doesn't
echo -n > "$output"

# Add a header to the Markdown file listing the directories
echo "# Files from directories: $argv" >> "$output"
echo "" >> "$output"

# Use fd to find all files in the specified directories
# -0 ensures null-terminated output to handle filenames with newlines
fd --type f -0 . $argv | while read -l -z file
    # Write the filepath as a Markdown heading
    echo "### $file" >> "$output"
    # Start the code block
    echo "``````" >> "$output"
    # Append the file's content
    cat "$file" >> "$output"
    # End the code block
    echo "``````" >> "$output"
    # Add a blank line for separation
    echo "" >> "$output"
end

# Check if fd encountered an error
if test $pipestatus[1] -ne 0
    echo "Error running fd"
    exit 1
end

# Notify the user of successful completion
echo "Markdown file generated: $output"