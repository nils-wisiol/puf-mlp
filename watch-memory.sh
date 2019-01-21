#!/bin/bash -e

python aseeri.py &
#python mlp.py &
psrecord $! --interval .05 --plot plot.png
#xdg-open plot.png
