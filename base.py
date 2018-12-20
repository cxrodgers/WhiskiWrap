"""Main functions for running input videos through trace.

The overall algorithm is contained in `interleaved_reading_and_tracing`. 
* The input can be a video file or a directory of PF files.
* Chunks of ~200 frames are read using ffmpeg, and then written to disk
  as uncompressed tiff stacks.
* Trace is called in parallel on each tiff stack
* Additional chunks are read as trace completes.
* At the end, all of the HDF5 files are stitched together.

The previous function `pipeline_trace` is now deprecated.
"""

try:
    import tifffile
except ImportError:
    pass
import os
import numpy as np
import subprocess
import multiprocessing
import tables
try:
    from whisk.python import trace
    from whisk.python.traj import MeasurementsTable
except ImportError:
    print "cannot import whisk"
import pandas
import WhiskiWrap
from WhiskiWrap import video_utils
import my
import scipy.io
import ctypes
import glob
import time
import shutil
import itertools

# Find the repo directory and the default param files
# The banks don't differe with sensitive or default
DIRECTORY = os.path.split(__file__)[0]
PARAMETERS_FILE = os.path.join(DIRECTORY, 'default.parameters')
SENSITIVE_PARAMETERS_FILE = os.path.join(DIRECTORY, 'sensitive.parameters')
HALFSPACE_DB_FILE = os.path.join(DIRECTORY, 'halfspace.detectorbank')
LINE_DB_FILE = os.path.join(DIRECTORY, 'line.detectorbank')

# libpfDoubleRate library, needed for PFReader
LIB_DOUBLERATE = os.path.join(DIRECTORY, 'libpfDoubleRate.so')

def copy_parameters_files(target_directory, sensitive=False):
    """Copies in parameters and banks"""
    if sensitive:
        shutil.copyfile(SENSITIVE_PARAMETERS_FILE, os.path.join(target_directory,
            'default.parameters'))
    else:
        shutil.copyfile(PARAMETERS_FILE, os.path.join(target_directory,
            'default.parameters'))
    
    # Banks are the same regardless
    shutil.copyfile(HALFSPACE_DB_FILE, os.path.join(target_directory,
        'halfspace.detectorbank'))
    shutil.copyfile(LINE_DB_FILE, os.path.join(target_directory,
        'line.detectorbank'))

class WhiskerSeg(tables.IsDescription):
    time = tables.UInt32Col()
    id = tables.UInt16Col()
    tip_x = tables.Float32Col()
    tip_y = tables.Float32Col()
    fol_x = tables.Float32Col()
    fol_y = tables.Float32Col()
    pixlen = tables.UInt16Col()
    chunk_start = tables.UInt32Col()

class WhiskerSeg_measure(tables.IsDescription):
    time = tables.UInt32Col()
    id = tables.UInt16Col()
    tip_x = tables.Float32Col()
    tip_y = tables.Float32Col()
    fol_x = tables.Float32Col()
    fol_y = tables.Float32Col()
    pixlen = tables.UInt16Col()
    length = tables.Float32Col()
    score = tables.Float32Col()
    angle = tables.Float32Col()
    curvature = tables.Float32Col()
    chunk_start = tables.UInt32Col()


def write_chunk(chunk, chunkname, directory='.'):
    tifffile.imsave(os.path.join(directory, chunkname), chunk, compress=0)

def trace_chunk(video_filename, delete_when_done=False):
    """Run trace on an input file
    
    First we create a whiskers filename from `video_filename`, which is
    the same file with '.whiskers' replacing the extension. Then we run
    trace using subprocess.
    
    Care is taken to move into the working directory during trace, and then
    back to the original directory.
    
    Returns:
        stdout, stderr
    """
    print "Starting", video_filename
    orig_dir = os.getcwd()
    run_dir, raw_video_filename = os.path.split(os.path.abspath(video_filename))
    whiskers_file = WhiskiWrap.utils.FileNamer.from_video(video_filename).whiskers
    command = ['trace', raw_video_filename, whiskers_file]

    os.chdir(run_dir)
    try:
        pipe = subprocess.Popen(command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        stdout, stderr = pipe.communicate()  
    except:
        raise
    finally:
        os.chdir(orig_dir)
    print "Done", video_filename
    
    if not os.path.exists(whiskers_file):
        print raw_video_filename
        raise IOError("tracing seems to have failed")

    if delete_when_done:
        os.remove(video_filename)
    
    return {'video_filename': video_filename, 'stdout': stdout, 'stderr': stderr}

def measure_chunk(whiskers_filename, face, delete_when_done=False):
    """Run measure on an input file
    
    First we create a measurement filename from `whiskers_filename`, which is
    the same file with '.measurements' replacing the extension. Then we run
    trace using subprocess.
    
    Care is taken to move into the working directory during trace, and then
    back to the original directory.
    
    Returns:
        stdout, stderr
    """
    print "Starting", whiskers_filename
    orig_dir = os.getcwd()
    run_dir, raw_whiskers_filename = os.path.split(os.path.abspath(whiskers_filename))
    measurements_file = WhiskiWrap.utils.FileNamer.from_whiskers(whiskers_filename).measurements
    command = ['measure', '--face', face, raw_whiskers_filename, measurements_file]

    os.chdir(run_dir)
    try:
        pipe = subprocess.Popen(command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        stdout, stderr = pipe.communicate()  
    except:
        raise
    finally:
        os.chdir(orig_dir)
    print "Done", whiskers_filename
    
    if not os.path.exists(measurements_file):
        print raw_whiskers_filename
        raise IOError("measurement seems to have failed")

    if delete_when_done:
        os.remove(whiskers_filename)
    
    return {'whiskers_filename': whiskers_filename, 'stdout': stdout, 'stderr': stderr}

def sham_trace_chunk(video_filename):
    print "sham tracing", video_filename
    time.sleep(2)
    return video_filename

def setup_hdf5(h5_filename, expectedrows, measure=False):

    # Open file
    h5file = tables.open_file(h5_filename, mode="w")    

    if not measure:
        WhiskerDescription = WhiskerSeg
    elif measure:
        WhiskerDescription = WhiskerSeg_measure
    
    # A group for the normal data
    table = h5file.create_table(h5file.root, "summary", WhiskerDescription, 
        "Summary data about each whisker segment",
        expectedrows=expectedrows)

    # Put the contour here
    xpixels_vlarray = h5file.create_vlarray(
        h5file.root, 'pixels_x', 
        tables.Float32Atom(shape=()),
        title='Every pixel of each whisker (x-coordinate)',
        expectedrows=expectedrows)
    ypixels_vlarray = h5file.create_vlarray(
        h5file.root, 'pixels_y', 
        tables.Float32Atom(shape=()),
        title='Every pixel of each whisker (y-coordinate)',
        expectedrows=expectedrows)
    
    h5file.close()
    
def append_whiskers_to_hdf5(whisk_filename, h5_filename, chunk_start, measurements_filename=None):
    """Load data from whisk_file and put it into an hdf5 file
    
    The HDF5 file will have two basic components:
        /summary : A table with the following columns:
            time, id, fol_x, fol_y, tip_x, tip_y, pixlen
            These are all directly taken from the whisk file
        /pixels_x : A vlarray of the same length as summary but with the
            entire array of x-coordinates of each segment.
        /pixels_y : Same but for y-coordinates
    """
    ## Load it, so we know what expectedrows is
    # This loads all whisker info into C data types
    # wv is like an array of trace.LP_cWhisker_Seg
    # Each entry is a trace.cWhisker_Seg and can be converted to
    # a python object via: wseg = trace.Whisker_Seg(wv[idx])
    # The python object responds to .time and .id (integers) and .x and .y (numpy
    # float arrays).
    #wv, nwhisk = trace.Debug_Load_Whiskers(whisk_filename)
    print whisk_filename
    
    whiskers = trace.Load_Whiskers(whisk_filename)
    nwhisk = np.sum(map(len, whiskers.values()))

    if measurements_filename is not None:
        print measurements_filename
        M = MeasurementsTable(str(measurements_filename))
        measurements = M.get_shape_table()
        measurements_idx = 0

    # Open file
    h5file = tables.open_file(h5_filename, mode="a")

    ## Iterate over rows and store
    table = h5file.get_node('/summary')
    h5seg = table.row
    xpixels_vlarray = h5file.get_node('/pixels_x')
    ypixels_vlarray = h5file.get_node('/pixels_y')
    for frame, frame_whiskers in whiskers.iteritems():
        for whisker_id, wseg in frame_whiskers.iteritems():
            # Write to the table
            h5seg['chunk_start'] = chunk_start
            h5seg['time'] = wseg.time + chunk_start
            h5seg['id'] = wseg.id
            h5seg['fol_x'] = wseg.x[0]
            h5seg['fol_y'] = wseg.y[0]
            h5seg['tip_x'] = wseg.x[-1]
            h5seg['tip_y'] = wseg.y[-1]

            if measurements_filename is not None:
                h5seg['length'] = measurements[measurements_idx][0]
                h5seg['score'] = measurements[measurements_idx][1]
                h5seg['angle'] = measurements[measurements_idx][2]
                h5seg['curvature'] = measurements[measurements_idx][3]
                h5seg['pixlen'] = len(wseg.x)
                measurements_idx += 1
           
            assert len(wseg.x) == len(wseg.y)
            h5seg.append()
            
            # Write x
            xpixels_vlarray.append(wseg.x)
            ypixels_vlarray.append(wseg.y)
    

    table.flush()
    h5file.close()    

def pipeline_trace(input_vfile, h5_filename,
    epoch_sz_frames=3200, chunk_sz_frames=200, 
    frame_start=0, frame_stop=None,
    n_trace_processes=4, expectedrows=1000000, flush_interval=100000,
    measure=False,face='right'):
    """Trace a video file using a chunked strategy.
    
    This is now deprecated in favor of interleaved_reading_and_tracing.
    The issue with this function is that it has to write out all of the tiffs
    first, before tracing, which is a wasteful use of disk space.
    
    input_vfile : input video filename
    h5_filename : output HDF5 file
    epoch_sz_frames : Video is first broken into epochs of this length
    chunk_sz_frames : Each epoch is broken into chunks of this length
    frame_start, frame_stop : where to start and stop processing
    n_trace_processes : how many simultaneous processes to use for tracing
    expectedrows, flush_interval : used to set up hdf5 file
    
    TODO: combine the reading and writing stages using frame_func so that
    we don't have to load the whole epoch in at once. In fact then we don't
    even need epochs at all.
    """
    WhiskiWrap.utils.probe_needed_commands()
    
    # Figure out where to store temporary data
    input_vfile = os.path.abspath(input_vfile)
    input_dir = os.path.split(input_vfile)[0]    

    # Setup the result file
    setup_hdf5(h5_filename, expectedrows, measure=measure)

    # Figure out how many frames and epochs
    duration = my.video.get_video_duration2(input_vfile)
    frame_rate = my.video.get_video_params(input_vfile)[2]
    total_frames = int(np.rint(duration * frame_rate))
    if frame_stop is None:
        frame_stop = total_frames
    if frame_stop > total_frames:
        print "too many frames requested, truncating"
        frame_stop = total_frames
    
    # Iterate over epochs
    for start_epoch in range(frame_start, frame_stop, epoch_sz_frames):
        # How many frames in this epoch
        stop_epoch = np.min([frame_stop, start_epoch + epoch_sz_frames])
        print "Epoch %d - %d" % (start_epoch, stop_epoch)
        
        # Chunks
        chunk_starts = np.arange(start_epoch, stop_epoch, chunk_sz_frames)
        chunk_names = ['chunk%08d.tif' % nframe for nframe in chunk_starts]
        whisk_names = ['chunk%08d.whiskers' % nframe for nframe in chunk_starts]
        

        # read everything
        # need to be able to crop here
        print "Reading"
        frames = video_utils.process_chunks_of_video(input_vfile, 
            frame_start=start_epoch, frame_stop=stop_epoch,
            frames_per_chunk=chunk_sz_frames, # only necessary for chunk_func
            frame_func=None, chunk_func=None,
            verbose=False, finalize='listcomp')

        # Dump frames into tiffs or lossless
        print "Writing"
        for n_whiski_chunk, chunk_name in enumerate(chunk_names):
            print n_whiski_chunk
            chunkstart = n_whiski_chunk * chunk_sz_frames
            chunkstop = (n_whiski_chunk + 1) * chunk_sz_frames
            chunk = frames[chunkstart:chunkstop]
            if len(chunk) in [3, 4]:
                print "WARNING: trace will fail on tiff stacks of length 3 or 4"
            write_chunk(chunk, chunk_name, input_dir)
        
        # Also write lossless and/or lossy monitor video here?
        # would really only be useful if cropping applied

        # trace each
        print "Tracing"
        pool = multiprocessing.Pool(n_trace_processes)        
        trace_res = pool.map(trace_chunk, 
            [os.path.join(input_dir, chunk_name)
                for chunk_name in chunk_names])
        pool.close()

        # take measurements:
        if measure:
            print "Measuring"
            pool = multiprocessing.Pool(n_trace_processes)
            meas_res = pool.map(measure_chunk_star, 
                itertools.izip([os.path.join(input_dir, whisk_name)
                    for whisk_name in whisk_names],itertools.repeat(face)))
            pool.close()
        

        # stitch
        print "Stitching"
        for chunk_start, chunk_name in zip(chunk_starts, chunk_names):
            # Append each chunk to the hdf5 file
            fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(
                os.path.join(input_dir, chunk_name))
            
            if not measure:
                append_whiskers_to_hdf5(
                    whisk_filename=fn.whiskers,
                    h5_filename=h5_filename, 
                    chunk_start=chunk_start)
            elif measure:
                append_whiskers_to_hdf5(
                    whisk_filename=fn.whiskers,
                    measurements_filename=fn.measurements,
                    h5_filename=h5_filename, 
                    chunk_start=chunk_start)



def write_video_as_chunked_tiffs(input_reader, tiffs_to_trace_directory,
    chunk_size=200, chunk_name_pattern='chunk%08d.tif',
    stop_after_frame=None, monitor_video=None, timestamps_filename=None,
    monitor_video_kwargs=None):
    """Write frames to disk as tiff stacks
    
    input_reader : object providing .iter_frames() method and perhaps
        also a .timestamps attribute. For instance, PFReader, or some
        FFmpegReader object.
    tiffs_to_trace_directory : where to store the chunked tiffs
    stop_after_frame : to stop early
    monitor_video : if not None, should be a filename to write a movie to
    timestamps_filename : if not None, should be the name to write timestamps
    monitor_video_kwargs : ffmpeg params
    
    Returns: ChunkedTiffWriter object    
    """
    # Tiff writer
    ctw = WhiskiWrap.ChunkedTiffWriter(tiffs_to_trace_directory,
        chunk_size=chunk_size, chunk_name_pattern=chunk_name_pattern)

    # FFmpeg writer is initalized after first frame
    ffw = None

    # Iterate over frames
    for nframe, frame in enumerate(input_reader.iter_frames()):
        # Stop early?
        if stop_after_frame is not None and nframe >= stop_after_frame:
            break
        
        # Write to chunked tiff
        ctw.write(frame)
        
        # Optionally write to monitor video
        if monitor_video is not None:
            # Initialize ffw after first frame so we know the size
            if ffw is None:
                ffw = WhiskiWrap.FFmpegWriter(monitor_video, 
                    frame_width=frame.shape[1], frame_height=frame.shape[0],
                    **monitor_video_kwargs)
            ffw.write(frame)

    # Finalize writers
    ctw.close()
    if ffw is not None:
        ff_stdout, ff_stderr = ffw.close()

    # Also write timestamps as numpy file
    if hasattr(input_reader, 'timestamps') and timestamps_filename is not None:
        timestamps = np.concatenate(input_reader.timestamps)
        assert len(timestamps) >= ctw.frames_written
        np.save(timestamps_filename, timestamps[:ctw.frames_written])

    return ctw

def trace_chunked_tiffs(input_tiff_directory, h5_filename,
    n_trace_processes=4, expectedrows=1000000,
    ):
    """Trace tiffs that have been written to disk in parallel and stitch.
    
    input_tiff_directory : directory containing tiffs
    h5_filename : output HDF5 file
    n_trace_processes : how many simultaneous processes to use for tracing
    expectedrows : used to set up hdf5 file
    """
    WhiskiWrap.utils.probe_needed_commands()
    
    # Setup the result file
    setup_hdf5(h5_filename, expectedrows)
    
    # The tiffs have been written, figure out which they are
    tif_file_number_strings = my.misc.apply_and_filter_by_regex(
        '^chunk(\d+).tif$', os.listdir(input_tiff_directory), sort=False)
    tif_full_filenames = [
        os.path.join(input_tiff_directory, 'chunk%s.tif' % fns)
        for fns in tif_file_number_strings]
    tif_file_numbers = map(int, tif_file_number_strings)
    tif_ordering = np.argsort(tif_file_numbers)
    tif_sorted_filenames = np.array(tif_full_filenames)[
        tif_ordering]
    tif_sorted_file_numbers = np.array(tif_file_numbers)[
        tif_ordering]

    # trace each
    print "Tracing"
    pool = multiprocessing.Pool(n_trace_processes)        
    trace_res = pool.map(trace_chunk, tif_sorted_filenames)
    pool.close()
    
    # stitch
    print "Stitching"
    for chunk_start, chunk_name in zip(tif_sorted_file_numbers, tif_sorted_filenames):
        # Append each chunk to the hdf5 file
        fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(chunk_name)
        append_whiskers_to_hdf5(
            whisk_filename=fn.whiskers,
            h5_filename=h5_filename, 
            chunk_start=chunk_start)


def interleaved_reading_and_tracing(input_reader, tiffs_to_trace_directory,
    sensitive=False,
    chunk_size=200, chunk_name_pattern='chunk%08d.tif',
    stop_after_frame=None, delete_tiffs=True,
    timestamps_filename=None, monitor_video=None, 
    monitor_video_kwargs=None, write_monitor_ffmpeg_stderr_to_screen=False,
    h5_filename=None, frame_func=None,
    n_trace_processes=4, expectedrows=1000000,    
    verbose=True, skip_stitch=False,
    ):
    """Read, write, and trace each chunk, one at a time.
    
    This is an alternative to first calling:
        write_video_as_chunked_tiffs
    And then calling
        trace_chunked_tiffs
    
    input_reader : Typically a PFReader or FFmpegReader
    tiffs_to_trace_directory : Location to write the tiffs
    sensitive: if False, use default. If True, lower MIN_SIGNAL
    chunk_size : frames per chunk
    chunk_name_pattern : how to name them
    stop_after_frame : break early, for debugging
    delete_tiffs : whether to delete tiffs after done tracing
    timestamps_filename : Where to store the timestamps
        Only vallid for PFReader input_reader
    monitor_video : filename for a monitor video
        If None, no monitor video will be written
    monitor_video_kwargs : kwargs to pass to FFmpegWriter for monitor
    write_monitor_ffmpeg_stderr_to_screen : whether to display
        output from ffmpeg writing instance
    h5_filename : hdf5 file to stitch whiskers information into
    frame_func : function to apply to each frame
        If 'invert', will apply 255 - frame
    n_trace_processes : number of simultaneous trace processes
    expectedrows : how to set up hdf5 file
    verbose : verbose
    skip_stitch : skip the stitching phase
    
    Returns: dict
        trace_pool_results : result of each call to trace
        monitor_ff_stderr, monitor_ff_stdout : results from monitor
            video ffmpeg instance
    """
    ## Set up kwargs
    if monitor_video_kwargs is None:
        monitor_video_kwargs = {}
    
    if frame_func == 'invert':
        frame_func = lambda frame: 255 - frame
    
    # Check commands
    WhiskiWrap.utils.probe_needed_commands()
    
    ## Initialize readers and writers
    if verbose:
        print "initalizing readers and writers"
    # Tiff writer
    ctw = WhiskiWrap.ChunkedTiffWriter(tiffs_to_trace_directory,
        chunk_size=chunk_size, chunk_name_pattern=chunk_name_pattern)

    # FFmpeg writer is initalized after first frame
    ffw = None

    # Setup the result file
    if not skip_stitch:
        setup_hdf5(h5_filename, expectedrows)
    
    # Copy the parameters files
    copy_parameters_files(tiffs_to_trace_directory, sensitive=sensitive)
    
    ## Set up the worker pool
    # Pool of trace workers
    trace_pool = multiprocessing.Pool(n_trace_processes)        
    
    # Keep track of results
    trace_pool_results = []
    deleted_tiffs = []
    def log_result(result):
        trace_pool_results.append(result)
    
    ## Iterate over chunks
    out_of_frames = False
    nframe = 0
    
    # Init the iterator outside of the loop so that it persists
    iter_obj = input_reader.iter_frames()
    
    while not out_of_frames:
        # Get a chunk of frames
        if verbose:
            print "loading chunk of frames starting with ", nframe
        chunk_of_frames = []
        for frame in iter_obj:
            if frame_func is not None:
                frame = frame_func(frame)
            chunk_of_frames.append(frame)
            nframe = nframe + 1
            if stop_after_frame is not None and nframe >= stop_after_frame:
                break
            if len(chunk_of_frames) == chunk_size:
                break
    
        # Check if we ran out
        if len(chunk_of_frames) != chunk_size:
            out_of_frames = True
                
        ## Write tiffs
        # We do this synchronously to ensure that it happens before
        # the trace starts
        for frame in chunk_of_frames:
            ctw.write(frame)        
        
        # Make sure the chunk was written, in case this is the last one
        # and we didn't reach chunk_size yet
        if len(chunk_of_frames) != chunk_size:
            ctw._write_chunk()
        assert ctw.count_unwritten_frames() == 0

        # Figure out which tiff file was just generated
        tif_filename = ctw.chunknames_written[-1]
        
        ## Start trace
        trace_pool.apply_async(trace_chunk, args=(tif_filename, delete_tiffs),
            callback=log_result)
        
        ## Determine whether we can delete any tiffs
        #~ if delete_tiffs:
            #~ tiffs_to_delete = [
                #~ tpres['video_filename'] for tpres in trace_pool_results
                #~ if tpres['video_filename'] not in deleted_tiffs]
            #~ for filename in tiffs_to_delete:
                #~ if verbose:
                    #~ print "deleting", filename
                #~ os.remove(filename)
    
        ## Start monitor encode
        # This is also synchronous, otherwise the input buffer might fill up
        if monitor_video is not None:        
            if ffw is None:
                ffw = WhiskiWrap.FFmpegWriter(monitor_video, 
                    frame_width=frame.shape[1], frame_height=frame.shape[0],
                    write_stderr_to_screen=write_monitor_ffmpeg_stderr_to_screen,
                    **monitor_video_kwargs)
            for frame in chunk_of_frames:
                ffw.write(frame)        
        
        ## Determine if we should pause
        while len(ctw.chunknames_written) > len(trace_pool_results) + 2 * n_trace_processes:
            print "waiting for tracing to catch up"
            time.sleep(30)
    
    ## Wait for trace to complete
    if verbose:
        print "done with reading and writing, just waiting for tracing"
    # Tell it no more jobs, so close when done
    trace_pool.close()
    
    # Wait for everything to finish
    trace_pool.join()
    
    ## Error check the tifs that were processed
    # Get the tifs we wrote, and the tifs we trace
    written_chunks = sorted(ctw.chunknames_written)
    traced_filenames = sorted([
        res['video_filename'] for res in trace_pool_results])
    
    # Check that they are the same
    if not np.all(np.array(written_chunks) == np.array(traced_filenames)):
        raise ValueError("not all chunks were traced")

    ## Extract the chunk numbers from the filenames
    # The tiffs have been written, figure out which they are
    split_traced_filenames = [os.path.split(fn)[1] for fn in traced_filenames]
    tif_file_number_strings = my.misc.apply_and_filter_by_regex(
        '^chunk(\d+).tif$', split_traced_filenames, sort=False)
    tif_full_filenames = [
        os.path.join(tiffs_to_trace_directory, 'chunk%s.tif' % fns)
        for fns in tif_file_number_strings]
    tif_file_numbers = map(int, tif_file_number_strings)
    tif_ordering = np.argsort(tif_file_numbers)
    tif_sorted_filenames = np.array(tif_full_filenames)[
        tif_ordering]
    tif_sorted_file_numbers = np.array(tif_file_numbers)[
        tif_ordering]
    
    # stitch
    if not skip_stitch:
        print "Stitching"
        zobj = zip(tif_sorted_file_numbers, tif_sorted_filenames)
        for chunk_start, chunk_name in zobj:
            # Append each chunk to the hdf5 file
            fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(chunk_name)
            append_whiskers_to_hdf5(
                whisk_filename=fn.whiskers,
                h5_filename=h5_filename, 
                chunk_start=chunk_start)

    # Finalize writers
    ctw.close()
    if ffw is not None:
        ff_stdout, ff_stderr = ffw.close()
    else:
        ff_stdout, ff_stderr = None, None

    # Also write timestamps as numpy file
    if hasattr(input_reader, 'timestamps') and timestamps_filename is not None:
        timestamps = np.concatenate(input_reader.timestamps)
        assert len(timestamps) >= ctw.frames_written
        np.save(timestamps_filename, timestamps[:ctw.frames_written])

    return {'trace_pool_results': trace_pool_results,
        'monitor_ff_stdout': ff_stdout,
        'monitor_ff_stderr': ff_stderr,
        'tif_sorted_file_numbers': tif_sorted_file_numbers,
        'tif_sorted_filenames': tif_sorted_filenames,
        }

def compress_pf_to_video(input_reader, chunk_size=200, stop_after_frame=None,
    timestamps_filename=None, monitor_video=None, monitor_video_kwargs=None, 
    write_monitor_ffmpeg_stderr_to_screen=False, frame_func=None, verbose=True,
    ):
    """Read modulated data and compress to video
    
    Adapted from interleaved_reading_and_tracing
    
    input_reader : typically a PFReader
    chunk_size : frames per chunk
    stop_after_frame : break early, for debugging
    timestamps_filename : Where to store the timestamps
        Only valid for PFReader input_reader
    monitor_video : filename for a monitor video
        If None, no monitor video will be written
    monitor_video_kwargs : kwargs to pass to FFmpegWriter for monitor
        If None, the default is {'qp': 15} for a high-fidelity compression
        that is still ~6x smaller than lossless.
    write_monitor_ffmpeg_stderr_to_screen : whether to display
        output from ffmpeg writing instance
    frame_func : function to apply to each frame
        If 'invert', will apply 255 - frame
    verbose : verbose
    
    Returns: dict
        monitor_ff_stderr, monitor_ff_stdout : results from monitor
            video ffmpeg instance
    """
    ## Set up kwargs
    if monitor_video_kwargs is None:
        monitor_video_kwargs = {'qp': 15}
    
    if frame_func == 'invert':
        frame_func = lambda frame: 255 - frame
    
    ## Initialize readers and writers
    if verbose:
        print "initalizing readers and writers"

    # FFmpeg writer is initalized after first frame
    ffw = None

    ## Iterate over chunks
    out_of_frames = False
    nframe = 0
    nframes_written = 0
    
    # Init the iterator outside of the loop so that it persists
    iter_obj = input_reader.iter_frames()
    
    while not out_of_frames:
        # Get a chunk of frames
        if verbose:
            print "loading chunk of frames starting with ", nframe
        chunk_of_frames = []
        for frame in iter_obj:
            if frame_func is not None:
                frame = frame_func(frame)
            chunk_of_frames.append(frame)
            nframe = nframe + 1
            if stop_after_frame is not None and nframe >= stop_after_frame:
                break
            if len(chunk_of_frames) == chunk_size:
                break
    
        # Check if we ran out
        if len(chunk_of_frames) != chunk_size:
            out_of_frames = True
    
        ## Start monitor encode
        # This is also synchronous, otherwise the input buffer might fill up
        if monitor_video is not None:        
            if ffw is None:
                ffw = WhiskiWrap.FFmpegWriter(monitor_video, 
                    frame_width=frame.shape[1], frame_height=frame.shape[0],
                    write_stderr_to_screen=write_monitor_ffmpeg_stderr_to_screen,
                    **monitor_video_kwargs)
            for frame in chunk_of_frames:
                ffw.write(frame)    
                nframes_written = nframes_written + 1
    
    # Finalize writers
    if ffw is not None:
        ff_stdout, ff_stderr = ffw.close()
    else:
        ff_stdout, ff_stderr = None, None

    # Also write timestamps as numpy file
    if hasattr(input_reader, 'timestamps') and timestamps_filename is not None:
        timestamps = np.concatenate(input_reader.timestamps)

        # These assertions only make sense if we wrote the whole file
        if stop_after_frame is None:
            assert len(timestamps) == nframes_written
            assert nframes_written == nframe
        
        # Save timestamps
        np.save(timestamps_filename, timestamps)

    return {
        'monitor_ff_stdout': ff_stdout,
        'monitor_ff_stderr': ff_stderr,
    }
    

class PFReader:
    """Reads photonfocus modulated data stored in matlab files"""
    def __init__(self, input_directory, n_threads=4, verbose=True, 
        error_on_unsorted_filetimes=True):
        """Initialize a new reader.
        
        input_directory : where the mat files are
            They are assumed to have a format like img10.mat, etc.
            They should contain variables called 'img' (a 4d array of
            modulated frames) and 't' (timestamps of each frame).
        n_threads : sent to pfDoubleRate_SetNrOfThreads
        
        error_on_unsorted_filetimes : bool
            Whether to raise an error if the modification times of the 
            matfiles are not in sorted order, which typically happens if
            something has gone wrong (but could just be that the file times
            weren't preserved)
        """
        self.input_directory = input_directory
        self.verbose = verbose

        ## Load the libraries
        # boost_thread needs boost_system
        # I think it used to be able to find boost_system without this line
        libboost_system = ctypes.cdll.LoadLibrary(
            '/usr/local/lib/libboost_system.so.1.50.0')
        
        # Load boost_thread
        libboost_thread = ctypes.cdll.LoadLibrary(
            '/usr/local/lib/libboost_thread.so')
        
        # Load the pf_lib (which requires boost_thread)
        self.pf_lib = ctypes.cdll.LoadLibrary(LIB_DOUBLERATE)
        self.demod_func = self.pf_lib['pfDoubleRate_DeModulateImage']

        # Set the number of threads
        self.pf_lib['pfDoubleRate_SetNrOfThreads'](n_threads)
        
        # Find all the imgN.mat files in the input directory
        self.matfile_number_strings = my.misc.apply_and_filter_by_regex(
            '^img(\d+)\.mat$', os.listdir(self.input_directory), sort=False)
        self.matfile_names = [
            os.path.join(self.input_directory, 'img%s.mat' % fns)
            for fns in self.matfile_number_strings]
        self.matfile_numbers = map(int, self.matfile_number_strings)
        self.matfile_ordering = np.argsort(self.matfile_numbers)

        # Sort the names and numbers
        self.sorted_matfile_names = np.array(self.matfile_names)[
            self.matfile_ordering]
        self.sorted_matfile_numbers = np.array(self.matfile_numbers)[
            self.matfile_ordering]

        # Error check the file times
        filetimes = np.array([
            my.misc.get_file_time(filename)
            for filename in self.sorted_matfile_names])
        if (np.diff(filetimes) < 0).any():
            if error_on_unsorted_filetimes:
                raise IOError("unsorted matfiles")
            else:
                print "warning: unsorted matfiles"
        
        # Create variable to store timestamps
        self.timestamps = []
        self.n_frames_read = 0
        
        # Set these values once they are read from the file
        self.frame_height = None
        self.frame_width = None
    
    def iter_frames(self):
        """Yields frames as they are read and demodulated.
        
        Iterates through the matfiles in order, demodulates each frame,
        and yields them one at a time.
        
        The chunk of timestamps from each matfile is appended to the list
        self.timestamps, so that self.timestamps is a list of arrays. There
        will be more timestamps than read frames until the end of the chunk.
        
        Also sets self.frame_height and self.frame_width and checks that
        they are consistent over the session.
        """
        # Iterate through matfiles and load
        for matfile_name in self.sorted_matfile_names:
            if self.verbose:
                print "loading %s" % matfile_name
            
            # Load the raw data
            # This is the slowest step
            matfile_load = scipy.io.loadmat(matfile_name)
            matfile_t = matfile_load['t'].flatten()
            matfile_modulated_data = matfile_load['img'].squeeze()
            assert matfile_modulated_data.ndim == 3 # height, width, time

            # Append the timestamps
            self.timestamps.append(matfile_t)

            # Extract shape
            n_frames = len(matfile_t)
            assert matfile_modulated_data.shape[-1] == n_frames
            modulated_frame_width = matfile_modulated_data.shape[1]
            frame_height = matfile_modulated_data.shape[0]
            
            if self.verbose:
                print "loaded %d modulated frames @ %dx%d" % (n_frames,
                    modulated_frame_width, frame_height)
            
            # Find the demodulated width
            # This can be done by pfDoubleRate_GetDeModulatedWidth
            # but I don't understand what it's doing.
            if modulated_frame_width == 416:
                demodulated_frame_width = 800
            elif modulated_frame_width == 332:
                demodulated_frame_width = 640
            else:
                raise ValueError("unknown modulated width: %d" %
                    modulated_frame_width)
            
            # Store the frame sizes as necessary
            if self.frame_width is None:
                self.frame_width = demodulated_frame_width
            if self.frame_width != demodulated_frame_width:
                raise ValueError("inconsistent frame widths")
            if self.frame_height is None:
                self.frame_height = frame_height
            if self.frame_height != frame_height:
                raise ValueError("inconsistent frame heights")            
            
            # Create a buffer for the result of each frame
            demodulated_frame_buffer = ctypes.create_string_buffer(
                frame_height * demodulated_frame_width)
            
            # Iterate over frames
            for n_frame in range(n_frames):
                if self.verbose and np.mod(n_frame, 200) == 0:
                    print "iterator has reached frame %d" % n_frame
                
                # Convert image to char array
                # Ideally we would use a buffer here instead of a copy in order
                # to save time. But Matlab data comes back in Fortran order 
                # instead of C order, so this is not possible.
                frame_charry = ctypes.c_char_p(
                    matfile_modulated_data[:, :, n_frame].tobytes())
                #~ frame_charry = ctypes.c_char_p(
                    #~ bytes(matfile_modulated_data[:, :, n_frame].data))
                
                # Run
                self.demod_func(demodulated_frame_buffer, frame_charry,
                    demodulated_frame_width, frame_height, 
                    modulated_frame_width)
                
                # Extract the result from the buffer
                demodulated_frame = np.fromstring(demodulated_frame_buffer,
                    dtype=np.uint8).reshape(
                    frame_height, demodulated_frame_width)
                
                self.n_frames_read = self.n_frames_read + 1
                
                yield demodulated_frame
        
        if self.verbose:
            print "iterator is empty"
    
    def close(self):
        """Currently does nothing"""
        pass
    
    def isclosed(self):
        return True

class ChunkedTiffWriter:
    """Writes frames to a series of tiff stacks"""
    def __init__(self, output_directory, chunk_size=200,
        chunk_name_pattern='chunk%08d.tif'):
        """Initialize a new chunked tiff writer.
        
        output_directory : where to write the chunks
        chunk_size : frames per chunk
        chunk_name_pattern : how to name the chunk, using the number of
            the first frame in it
        """
        self.output_directory = output_directory
        self.chunk_size = chunk_size
        self.chunk_name_pattern = chunk_name_pattern
        
        # Initialize counters so we know what frame and chunk we're on
        self.frames_written = 0
        self.frame_buffer = []
        self.chunknames_written = []
    
    def write(self, frame):
        """Buffered write frame to tiff stacks"""
        # Append to buffer
        self.frame_buffer.append(frame)
        
        # Write chunk if buffer is full
        if len(self.frame_buffer) == self.chunk_size:
            self._write_chunk()

    def _write_chunk(self):
        if len(self.frame_buffer) != 0:
            # Form the chunk
            chunk = np.array(self.frame_buffer)
            
            # Name it
            chunkname = os.path.join(self.output_directory,
                self.chunk_name_pattern % self.frames_written)
            
            # Write it
            tifffile.imsave(chunkname, chunk, compress=0)
            
            # Update the counter
            self.frames_written += len(self.frame_buffer)
            self.frame_buffer = []        
            
            # Update the list of written chunks
            self.chunknames_written.append(chunkname)
    
    def count_unwritten_frames(self):
        """Returns the number of buffered, unwritten frames"""
        return len(self.frame_buffer)
    
    def close(self):
        """Finish writing any final unfinished chunk"""
        self._write_chunk()

class FFmpegReader:
    """Reads frames from a video file using ffmpeg process"""
    def __init__(self, input_filename, pix_fmt='gray', bufsize=10**9,
        duration=None, start_frame_time=None, start_frame_number=None,
        write_stderr_to_screen=False, vsync='drop'):
        """Initialize a new reader
        
        input_filename : name of file
        pix_fmt : used to format the raw data coming from ffmpeg into
            a numpy array
        bufsize : probably not necessary because we read one frame at a time
        duration : duration of video to read (-t parameter)
        start_frame_time, start_frame_number : -ss parameter
            Parsed using my.video.ffmpeg_frame_string
        write_stderr_to_screen : if True, writes to screen, otherwise to
            /dev/null
        """
        self.input_filename = input_filename
    
        # Get params
        self.frame_width, self.frame_height, self.frame_rate = \
            my.video.get_video_params(input_filename)
        
        # Set up pix_fmt
        if pix_fmt == 'gray':
            self.bytes_per_pixel = 1
        elif pix_fmt == 'rgb24':
            self.bytes_per_pixel = 3
        else:
            raise ValueError("can't handle pix_fmt:", pix_fmt)
        self.read_size_per_frame = self.bytes_per_pixel * \
            self.frame_width * self.frame_height
        
        # Create the command
        command = ['ffmpeg']
        
        # Add ss string
        if start_frame_time is not None or start_frame_number is not None:
            ss_string = my.video.ffmpeg_frame_string(input_filename,
                frame_time=start_frame_time, frame_number=start_frame_number)
            command += [
                '-ss', ss_string]
        
        command += [
            '-i', input_filename,
            '-vsync', vsync,
            '-f', 'image2pipe',
            '-pix_fmt', pix_fmt]
        
        # Add duration string
        if duration is not None:
            command += [
                '-t', str(duration),]
        
        # Add vcodec for pipe
        command += [
            '-vcodec', 'rawvideo', '-']
        
        # To store result
        self.n_frames_read = 0

        # stderr
        if write_stderr_to_screen:
            stderr = None
        else:
            stderr = open(os.devnull, 'w')

        # Init the pipe
        # We set stderr to null so it doesn't fill up screen or buffers
        # And we set stdin to PIPE to keep it from breaking our STDIN
        self.ffmpeg_proc = subprocess.Popen(command, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr, 
            bufsize=bufsize)

    def iter_frames(self):
        """Yields one frame at a time
        
        When done: terminates ffmpeg process, and stores any remaining
        results in self.leftover_bytes and self.stdout and self.stderr
        
        It might be worth writing this as a chunked reader if this is too
        slow. Also we need to be able to seek through the file.
        """
        # Read this_chunk, or as much as we can
        while(True):
            raw_image = self.ffmpeg_proc.stdout.read(self.read_size_per_frame)

            # check if we ran out of frames
            if len(raw_image) != self.read_size_per_frame:
                self.leftover_bytes = raw_image
                self.close()
                return
        
            # Convert to array
            flattened_im = np.fromstring(raw_image, dtype='uint8')
            if self.bytes_per_pixel == 1:
                frame = flattened_im.reshape(
                    (self.frame_height, self.frame_width))
            else:
                frame = flattened_im.reshape(
                    (self.frame_height, self.frame_width, self.bytes_per_pixel))

            # Update
            self.n_frames_read = self.n_frames_read + 1

            # Yield
            yield frame
    
    def close(self):
        """Closes the process"""
        # Need to terminate in case there is more data but we don't
        # care about it
        # But if it's already terminated, don't try to terminate again
        if self.ffmpeg_proc.returncode is None:
            self.ffmpeg_proc.terminate()
        
            # Extract the leftover bits
            self.stdout, self.stderr = self.ffmpeg_proc.communicate()
        
        return self.ffmpeg_proc.returncode
    
    def isclosed(self):
        if hasattr(self.ffmpeg_proc, 'returncode'):
            return self.ffmpeg_proc.returncode is not None
        else:
            # Never even ran? I guess this counts as closed.
            return True

class FFmpegWriter:
    """Writes frames to an ffmpeg compression process"""
    def __init__(self, output_filename, frame_width, frame_height,
        output_fps=30, vcodec='libx264', qp=15, preset='medium',
        input_pix_fmt='gray', output_pix_fmt='yuv420p', 
        write_stderr_to_screen=False):
        """Initialize the ffmpeg writer
        
        output_filename : name of output file
        frame_width, frame_height : Used to inform ffmpeg how to interpret
            the data coming in the stdin pipe
        output_fps : frame rate
        input_pix_fmt : Tell ffmpeg how to interpret the raw data on the pipe
            This should match the output generated by frame.tostring()
        output_pix_fmt : pix_fmt of the output
        crf : quality. 0 means lossless
        preset : speed/compression tradeoff
        write_stderr_to_screen :
            If True, writes ffmpeg's updates to screen
            If False, writes to /dev/null
        
        With old versions of ffmpeg (jon-severinsson) I was not able to get
        truly lossless encoding with libx264. It was clamping the luminances to
        16..235. Some weird YUV conversion? 
        '-vf', 'scale=in_range=full:out_range=full' seems to help with this
        In any case it works with new ffmpeg. Also the codec ffv1 will work
        but is slightly larger filesize.
        """
        # Open an ffmpeg process
        cmdstring = ('ffmpeg', 
            '-y', '-r', '%d' % output_fps,
            '-s', '%dx%d' % (frame_width, frame_height), # size of image string
            '-pix_fmt', input_pix_fmt,
            '-f', 'rawvideo',  '-i', '-', # raw video from the pipe
            '-pix_fmt', output_pix_fmt,
            '-vcodec', vcodec,
            '-qp', str(qp), 
            '-preset', preset,
            output_filename) # output encoding
        
        if write_stderr_to_screen:
            self.ffmpeg_proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE)
        else:
            self.ffmpeg_proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))       
    
    def write(self, frame):
        """Write a frame to the ffmpeg process"""
        self.ffmpeg_proc.stdin.write(frame.tostring())
    
    def write_bytes(self, bytestring):
        self.ffmpeg_proc.stdin.write(bytestring)
    
    def close(self):
        """Closes the ffmpeg process and returns stdout, stderr"""
        return self.ffmpeg_proc.communicate()


def measure_chunk_star(args):
    return measure_chunk(*args)


def read_whiskers_hdf5_summary(filename):
    """Reads and returns the `summary` table in an HDF5 file"""
    with tables.open_file(filename) as fi:
        summary = pandas.DataFrame.from_records(fi.root.summary.read())
    
    return summary
