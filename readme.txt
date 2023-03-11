In order to run the code install the required packages using the following command:

pip install -r requirements.txt

The code is written in python 3.8 and tested on macOS.

to run the script used to generate plots for the report run:

cd src

python maze.py

The code will generate the plots and save them in the figures folder.

To generate the benchmarking run the following command:

python benchmark.py

This will generate a csv file with the benchmarking results. Please note that this will take a while to run.
Moreover the results are slightly different from the ones reported in the report due to the random selection of the mazes and the polices for policy iteration.