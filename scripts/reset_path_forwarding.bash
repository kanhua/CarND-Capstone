#!/bin/bash

sudo cp nat.conf /Library/Preferences/VMware\ Fusion/vmnet8/nat.conf
sudo /Applications/VMware\ Fusion.app/Contents/Library/vmnet-cli --stop
sudo /Applications/VMware\ Fusion.app/Contents/Library/vmnet-cli --start