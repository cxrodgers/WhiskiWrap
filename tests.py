"""Functions for running tests and/or benchmarks with WhiskiWrap.

Each test is run in a session directory containing all the necessary
files for that test.

The function run_standard_benchmarks goes through a suite of tests. This
tests that everything is working properly and also serves as an example
of how to choose parameters and trace input videos.
"""
from __future__ import print_function


import os
import numpy as np
import tables
import time
import pandas
import WhiskiWrap
import shutil


def setup_session_directory(directory, input_video, force=False):
    """Create (or overwrite) directory for whisker tracking"""
    # Parse the input video filename
    input_video = os.path.abspath(os.path.expanduser(input_video))
    if not os.path.exists(input_video):
        raise ValueError("%s does not exist" % input_video)
    input_video_directory, input_video_filename = os.path.split(input_video)
    
    # Erase existing directory and create anew
    whiski_files = ['.mp4', '.avi', '.whiskers', '.tif', '.measurements',
        '.detectorbank', '.parameters', '.hdf5']
    if os.path.exists(directory):
        # Check that it looks like a whiskers directory
        file_list = os.listdir(directory)
        for filename in file_list:
            if (os.path.splitext(filename)[1]) not in whiski_files:
                raise ValueError(directory + 
                    " does not look safe to overwrite, aborting")
        
        # Get user confirmation
        if not force:
            confirm = raw_input('Ok to erase %s? [y/N]: ' % directory)
            if confirm.upper() != 'Y':
                raise ValueError("did not receive permission to setup test")
        
        # Erase
        os.system('rm -rf %s' % directory)
    os.mkdir(directory)
    
    # Copy the input video into the session directory
    new_video_filename = os.path.join(directory, input_video_filename)
    shutil.copyfile(input_video, new_video_filename)
    
    # Copy the parameter files in
    for filename in [WhiskiWrap.PARAMETERS_FILE, 
        WhiskiWrap.HALFSPACE_DB_FILE, WhiskiWrap.LINE_DB_FILE]:
        raw_filename = os.path.split(filename)[1]
        shutil.copyfile(filename, os.path.join(directory, raw_filename))
    
    return WhiskiWrap.utils.FileNamer.from_video(new_video_filename)

def run_benchmarks(benchmark_params, test_root, force=False):
    """Run the benchmarks
    
    For every row in benchmark params, run a trace on the input video
    using the params specified.
    
    benchmark_params: DataFrame with columns corresponding to keywords
        to pass to pipeline_trace. Should have columns 'name',
        'input_video', 'chunk_sz_frames', 'epoch_sz_frames',
        'frame_start', 'frame_stop', 'n_trace_processes', etc
    
    Returns:
        test_results, durations
        test_results : Dict from test['name'] to results read from hdf5 file
        durations : list of durations taken
    """
    WhiskiWrap.utils.probe_needed_commands()
    
    test_results = {}
    durations = []    
    for idx, test in benchmark_params.iterrows():
        print(test['name'])
        test_dir = os.path.expanduser(os.path.join(test_root, test['name']))
        fn = setup_session_directory(test_dir, test['input_video'], force=force)

        # Run
        start_time = time.time()
        WhiskiWrap.pipeline_trace(
            fn.video('mp4'),
            fn.hdf5,
            chunk_sz_frames=test['chunk_sz_frames'],
            epoch_sz_frames=test['epoch_sz_frames'],
            frame_start=test['frame_start'],
            frame_stop=test['frame_stop'],
            n_trace_processes=test['n_trace_processes'])
        stop_time = time.time()
        durations.append(stop_time - start_time)

        # Get the summary
        with tables.open_file(fn.hdf5) as fi:
            test_results[test['name']] = pandas.DataFrame.from_records(
                fi.root.summary.read()) 
    
    return test_results, durations

def run_standard_benchmarks(test_root='~/whiski_wrap_test', force=False,
    input_video=None, n_frames=None, epoch_sz=None,
    n_processes_l=(2, 4), chunk_size_l=(100, 300),
    ):
    """Run a suite of standard benchmarks.
    
    This function sets up a series of tests and then calls run_benchmarks
    on those tests.
    
    test_root : directory to store test results. Will be overwritten.
    force : if True, does not ask permission to overwrite anything
    input_video : which video to use as a test.
        Default: test_video165s.mp4 in the WhiskiWrap directory
    n_frames : number of frames to process
        Default: np.max(chunk_size_l) * np.max(n_process_l), that is, 
        just enough that the largest chunks will use all the processes
    epoch_sz : length of each epoch
        Default: equal to n_frames
    n_processes_l : a list of process numbers to test
    chunk_size_l : a list of chunk sizes to test
    
    The number of frames and the epoch length is chosen to be 
    np.max(chunk_size_l) * max_processes. It is assumed that additional
    epochs would take the same time.
    
    Returns: test_results, durations
    """
    # Check we have commands we need
    WhiskiWrap.utils.probe_needed_commands()
    
    # Set up test root
    test_root = normalize_path_and_optionally_get_permission(test_root,
        force=force)
    
    # Find the video to use
    if input_video is None:
        input_video = os.path.join(WhiskiWrap.DIRECTORY, 'test_video_165s.mp4')
    
    # Determine number of frames
    if n_frames is None:
        # Enough so that we use up all the processes on the largest chunk
        n_frames = np.max(n_processes_l) * np.max(chunk_size_l)
    if epoch_sz is None:
        epoch_sz = n_frames    
    
    # Construct the tests
    tests = []
    for chunk_sz in chunk_size_l:
        for n_process in n_processes_l:
            test_name = '%d_frames_%d_chunksz_%d_procs' % (
                n_frames, chunk_sz, n_process)
            tests.append([test_name, input_video, 0, n_frames, epoch_sz,
                chunk_sz, n_process])
    tests_df = pandas.DataFrame(tests, columns=(
        'name', 'input_video', 'frame_start', 'frame_stop', 
        'epoch_sz_frames', 'chunk_sz_frames', 'n_trace_processes'))

    # Run the tests
    test_results, durations = run_benchmarks(tests_df, test_root)
    tests_df['duration'] = durations

    return test_results, tests_df

def run_offset_test(test_root='~/whiski_wrap_test', start=1525, offset=5,
    n_frames=30, force=False):
    """Run a test where we offset the frame start"""
    # Check we have commands we need
    WhiskiWrap.utils.probe_needed_commands()
    
    # Set up test root
    test_root = normalize_path_and_optionally_get_permission(test_root,
        force=force)
    
    # Find the video to use
    vfile1 = os.path.join(WhiskiWrap.DIRECTORY, 'test_video2.mp4')

    # Construct the tests
    stop = start + n_frames
    tests = pandas.DataFrame([
        ['one_chunk', vfile1, start, stop, 100, 100, 1],
        ['one_chunk_offset', vfile1, start + offset, stop, 100, 100, 1],
        ],
        columns=(
            'name', 'input_video', 'frame_start', 'frame_stop', 
            'epoch_sz_frames', 'chunk_sz_frames', 'n_trace_processes'))
    
    # Run the tests
    test_results, durations = run_benchmarks(tests, test_root)

    return test_results, durations

def get_permission_for_test_root(test_root):
    """Ask for permission to run in test_root"""
    response = raw_input('Run tests in %s? [y/N]: ' % test_root)
    if response.upper() != 'Y':
        raise ValueError("did not receive permission to run test")   

def normalize_path_and_optionally_get_permission(test_root, force=False):
    # Form test root
    test_root = os.path.abspath(os.path.expanduser(test_root))
    
    # Get permission to use it
    if not force:
        get_permission_for_test_root(test_root)    
    
    return test_root

def run_standard(test_root='~/whiski_wrap_test', force=False):
    """Run a standard trace on a test file to get baseline time"""
    # Check we have commands we need
    WhiskiWrap.utils.probe_needed_commands()
    
    # Set up test root
    test_root = normalize_path_and_optionally_get_permission(test_root,
        force=force)
    
    # Find the video to use
    vfile1 = os.path.join(WhiskiWrap.DIRECTORY, 'test_video_165s.mp4')    

    # Set up the test directory
    fn = setup_session_directory(os.path.join(test_root, 'standard'), vfile1)

    # Run the test
    start_time = time.time()
    WhiskiWrap.trace_chunk(fn.video('mp4'))
    stop_time = time.time()
    standard_duration = stop_time - start_time
    
    # Stitch
    WhiskiWrap.setup_hdf5(fn.hdf5, expectedrows=100000)
    WhiskiWrap.append_whiskers_to_hdf5(
        whisk_filename=fn.whiskers,
        h5_filename=fn.hdf5, 
        chunk_start=0)    
    
    # Get the result
    with tables.open_file(fn.hdf5) as fi:
        test_result = pandas.DataFrame.from_records(
            fi.root.summary.read())     

    return test_result, standard_duration