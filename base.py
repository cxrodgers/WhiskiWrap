"""Main components of each stage of processing"""

import tifffile
import os
import numpy as np
import subprocess
import multiprocessing
import tables
try:
    from whisk.python import trace
except ImportError:
    print "cannot import whisk"
import pandas
import WhiskiWrap

# Find the repo directory and the default param files
DIRECTORY = os.path.split(__file__)[0]
PARAMETERS_FILE = os.path.join(DIRECTORY, 'default.parameters')
HALFSPACE_DB_FILE = os.path.join(DIRECTORY, 'halfspace.detectorbank')
LINE_DB_FILE = os.path.join(DIRECTORY, 'line.detectorbank')


class WhiskerSeg(tables.IsDescription):
    time = tables.UInt32Col()
    id = tables.UInt16Col()
    tip_x = tables.Float32Col()
    tip_y = tables.Float32Col()
    fol_x = tables.Float32Col()
    fol_y = tables.Float32Col()
    pixlen = tables.UInt16Col()
    chunk_start = tables.UInt16Col()

def write_chunk(chunk, chunkname, directory='.'):
    tifffile.imsave(os.path.join(directory, chunkname), chunk, compress=0)

def trace_chunk(video_filename):
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
    whiskers_file = FileNamer.from_video(video_filename).whiskers
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
        print raw_filename
        raise IOError("tracing seems to have failed")
    
    return stdout, stderr


def setup_hdf5(h5_filename, expectedrows):

    # Open file
    h5file = tables.open_file(h5_filename, mode="w")    
    
    
    # A group for the normal data
    table = h5file.create_table(h5file.root, "summary", WhiskerSeg, 
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
    
def append_whiskers_to_hdf5(whisk_filename, h5_filename, chunk_start):
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
    whiskers = trace.Load_Whiskers(whisk_filename)
    nwhisk = np.sum(map(len, whiskers.values()))

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
            h5seg['pixlen'] = len(wseg.x)
            assert len(wseg.x) == len(wseg.y)
            h5seg.append()
            
            # Write x
            xpixels_vlarray.append(wseg.x)
            ypixels_vlarray.append(wseg.y)

    table.flush()
    h5file.close()    

def pipeline_trace(input_vfile, h5_filename,
    epoch_sz_frames=100000, chunk_sz_frames=1000, 
    frame_start=0, frame_stop=None,
    n_trace_processes=4, expectedrows=1000000, flush_interval=100000,
    ):
    """Trace a video file using a chunked strategy.
    
    input_vfile : input video filename
    h5_filename : output HDF5 file
    epoch_sz_frames : Video is first broken into epochs of this length
    chunk_sz_frames : Each epoch is broken into chunks of this length
    frame_start, frame_stop : where to start and stop processing
    n_trace_processes : how many simultaneous processes to use for tracing
    expectedrows, flush_interval : used to set up hdf5 file
    """
    WhiskiWrap.utils.probe_needed_commands()
    
    # Figure out where to store temporary data
    input_vfile = os.path.abspath(input_vfile)
    input_dir = os.path.split(input_vfile)[0]    

    # Setup the result file
    setup_hdf5(h5_filename, expectedrows)

    # Figure out how many frames and epochs
    duration = WhiskiWrap.video_utils.get_video_duration(input_vfile)
    frame_rate = WhiskiWrap.video_utils.get_video_params(input_vfile)[2]
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
    
        # read everything
        # need to be able to crop here
        print "Reading"
        frames = WhiskiWrap.video_utils.process_chunks_of_video(input_vfile, 
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

        # stitch
        print "Stitching"
        for chunk_start, chunk_name in zip(chunk_starts, chunk_names):
            # Append each chunk to the hdf5 file
            fn = WhiskiWrap.utils.FileNamer.from_tiff_stack(
                os.path.join(input_dir, chunk_name))
            append_whiskers_to_hdf5(
                whisk_filename=fn.whiskers,
                h5_filename=h5_filename, 
                chunk_start=chunk_start)

            #~ # Try to put this in its own process so that it releases
            #~ # its memory leak upon completion
            #~ proc = multiprocessing.Process(
                #~ target=append_whiskers_to_hdf5,
                #~ kwargs={
                    #~ 'whisk_filename': os.path.join(
                        #~ input_dir, chunk_name + '.whiskers'),
                    #~ 'h5_filename': h5_filename,
                    #~ 'chunk_start': chunk_start})
            #~ proc.start()
            #~ proc.join()
            
            #~ if proc.exitcode != 0:
                #~ raise RuntimeError("some issue with stitching")



