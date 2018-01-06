#!/bin/sh

# This script extracts the velocity log file

grep "ctv" $HOME"/.ros/log/latest/dbw_node-4-stdout.log" > $HOME"/CarND-Capstone/velocity.log"


