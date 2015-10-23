"""Tests of functionality in WhiskiWrap"""


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

def run_benchmarks(benchmark_params, test_root):
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
        print test['name']
        test_dir = os.path.expanduser(os.path.join(test_root, test['name']))
        fn = setup_session_directory(test_dir, test['input_video'])

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

def run_standard_benchmarks(test_root='~/whiski_wrap_test', force=False):
    """Run a suite of standard benchmarks.
    
    Gets files from repo. Sets standard params.
    Calls run_benchmarks on the results
    """
    # Check we have commands we need
    WhiskiWrap.utils.probe_needed_commands()
    
    # Set up test root
    test_root = normalize_path_and_optionally_get_permission(test_root,
        force=force)
    
    # Find the video to use
    vfile1 = os.path.join(WhiskiWrap.DIRECTORY, 'test_video_165s.mp4')
    vfile2 = '/mnt/nas2/homes/chris/test_video_165s.mp4'
    
    # Construct the tests
    # NFRAMES must be at least NPROCS * CHUNKSZ or not all procs will be used
    # NFRAMES should be an even multiple of NPROCS * CHUNKSZ or some procs
    # will be left without work at the end.
    tests = pandas.DataFrame([
        ['one_chunk', vfile1, 0, 10, 100, 100, 1],
        ['one_chunk_offset', vfile1, 2, 10, 100, 100, 1],
        ['4800_frames_030_chunksz_04_procs_loc', vfile1, 0, 4800, 4800, 30, 4],
        ['4800_frames_030_chunksz_08_procs_loc', vfile1, 0, 4800, 4800, 30, 8],
        ['4800_frames_030_chunksz_16_procs_loc', vfile1, 0, 4800, 4800, 30, 16],
        ['4800_frames_100_chunksz_04_procs_loc', vfile1, 0, 4800, 4800, 100, 4],
        ['4800_frames_100_chunksz_08_procs_loc', vfile1, 0, 4800, 4800, 100, 8],
        ['4800_frames_100_chunksz_16_procs_loc', vfile1, 0, 4800, 4800, 100, 16],
        ['4800_frames_300_chunksz_04_procs_loc', vfile1, 0, 4800, 4800, 300, 4],
        ['4800_frames_300_chunksz_08_procs_loc', vfile1, 0, 4800, 4800, 300, 8],
        ['4800_frames_300_chunksz_16_procs_loc', vfile1, 0, 4800, 4800, 300, 16],
        ['4800_frames_030_chunksz_04_procs_nas', vfile2, 0, 4800, 4800, 30, 4],
        ['4800_frames_030_chunksz_08_procs_nas', vfile2, 0, 4800, 4800, 30, 8],
        ['4800_frames_030_chunksz_16_procs_nas', vfile2, 0, 4800, 4800, 30, 16],
        ['4800_frames_100_chunksz_04_procs_nas', vfile2, 0, 4800, 4800, 100, 4],
        ['4800_frames_100_chunksz_08_procs_nas', vfile2, 0, 4800, 4800, 100, 8],
        ['4800_frames_100_chunksz_16_procs_nas', vfile2, 0, 4800, 4800, 100, 16],
        ['4800_frames_300_chunksz_04_procs_nas', vfile2, 0, 4800, 4800, 300, 4],
        ['4800_frames_300_chunksz_08_procs_nas', vfile2, 0, 4800, 4800, 300, 8],
        ['4800_frames_300_chunksz_16_procs_nas', vfile2, 0, 4800, 4800, 300, 16],        
        ],
        columns=(
            'name', 'input_video', 'frame_start', 'frame_stop', 
            'epoch_sz_frames', 'chunk_sz_frames', 'n_trace_processes'))

    # Run the tests
    test_results, durations = run_benchmarks(tests, test_root)

    return test_results, durations

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
    vfile1 = os.path.join(WhiskiWrap.DIRECTORY, 'test_video2.mp4')    

    # Set up the test directory
    fn = setup_session_directory(os.path.join(test_root, 'standard'), vfile1)

    # Run the test
    start_time = time.time()
    trace_chunk(fn.video('mp4'))
    stop_time = time.time()
    standard_duration = stop_time - start_time
    
    # Stitch
    setup_hdf5(fn.hdf5, expectedrows=100000)
    append_whiskers_to_hdf5(
        whisk_filename=fn.whiskers,
        h5_filename=fn.hdf5, 
        chunk_start=0)    
    
    # Get the result
    with tables.open_file(fn.hdf5) as fi:
        test_result = pandas.DataFrame.from_records(
            fi.root.summary.read())     

    return test_result, standard_duration