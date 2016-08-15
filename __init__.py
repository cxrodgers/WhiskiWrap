"""WhiskiWrap provides tools for efficiently running whisk.

This module contains the following sub-modules:
    base - The basic functions for interacting with whisk. Everything is
        imported from base into the main WhiskiWrap namespace.
    tests - Benchmarks for running whiski
    utils - utility functions for dealing with files and programs on the
        system
    video_utils - functions for dealing with video files, usually via
        system calls to ffmpeg


Here is an example workflow:

import WhiskiWrap
import tables

# Run the benchmarks
test_results, durations = WhiskiWrap.tests.run_standard_benchmarks()

# Run on data
WhiskiWrap.pipeline_trace(
    input_video_filename='my_input_video.mp4',
    output_hdf5_filename='traced_whiskers.hdf5',
    chunk_sz_frames=300, epoch_sz_frames=3000, frame_start=0, frame_stop=6000,
    n_trace_processes=4)

# Load the results of that analysis
with tables.open_file('traced_whiskers.hdf5') as fi:
    test_results = pandas.DataFrame.from_records(fi.root.summary.read()) 
"""

import base
import tests
#import video_utils
import utils
reload(base)
from base import *