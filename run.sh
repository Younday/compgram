#!/bin/bash
# run the dependency parser
# if you run it without arguments you can see its options
python3 src/tinydepparser.py --train data/en.train --test data/en.dev --model model_dir --output en.dev.out