#!/usr/bin/python
import sys
import io

# Create the new config file for writing
config = io.open('nat.conf', 'w')

# Read the lines from the template, substitute the values, and write to the new config file
for line in io.open('nat_template.conf', 'r'):
    line = line.replace('${ip_name}', sys.argv[1])
    config.write(line)

# Close the files
config.close()
