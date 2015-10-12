import tifffile
import os
import my
import numpy as np
import subprocess
import multiprocessing
import tables
import trace

class WhiskerSeg(tables.IsDescription):
    time = tables.UInt32Col()
    id = tables.UInt16Col()
    tip_x = tables.Float32Col()
    tip_y = tables.Float32Col()
    fol_x = tables.Float32Col()
    fol_y = tables.Float32Col()
    pixlen = tables.UInt16Col()
    chunk_label = tables.UInt16Col()

def write_chunk(chunk, chunkname):
    tifffile.imsave(chunkname, chunk, compress=0)

def trace_chunk(chunk_name):
    print "Starting", chunk_name
    command = ['trace', chunk_name, chunk_name + '.whiskers']
    pipe = subprocess.Popen(command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    stdout, stderr = pipe.communicate()  
    print "Done", chunk_name


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
    
def append_whiskers_to_hdf5(whisk_filename, h5_filename, 
    chunk_label, verbose=True,
    flush_interval=100000, truncate_seg=None):
    """Load data from whisk_file and put it into an hdf5 file
    
    The HDF5 file will have two basic components:
        /summary : A table with the following columns:
            time, id, fol_x, fol_y, tip_x, tip_y, pixlen
            These are all directly taken from the whisk file
        /pixels_x : A vlarray of the same length as summary but with the
            entire array of x-coordinates of each segment.
        /pixels_y : Same but for y-coordinates
    
    truncate_seg : for debugging, stop after this many segments
    """
    ## Load it, so we know what expectedrows is
    # This loads all whisker info into C data types
    # wv is like an array of trace.LP_cWhisker_Seg
    # Each entry is a trace.cWhisker_Seg and can be converted to
    # a python object via: wseg = trace.Whisker_Seg(wv[idx])
    # The python object responds to .time and .id (integers) and .x and .y (numpy
    # float arrays).
    wv, nwhisk = trace.Debug_Load_Whiskers(whisk_filename)
    if truncate_seg is not None:
        nwhisk = truncate_seg

    # Open file
    h5file = tables.open_file(h5_filename, mode="a")

    ## Iterate over rows and store
    table = h5file.get_node('/summary')
    h5seg = table.row
    for idx in range(nwhisk):
        # Announce
        if verbose and np.mod(idx, 10000) == 0:
            print idx

        # Get the C object and convert to python
        # I suspect this is the bottleneck in speed
        cws = wv[idx]
        wseg = trace.Whisker_Seg(cws)

        # Write to the table
        h5seg['chunk_label'] = chunk_label
        h5seg['time'] = wseg.time
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

        if np.mod(idx, flush_interval) == 0:
            table.flush()

    table.flush()
    h5file.close()    


def pipeline_trace(input_vfile, h5_filename,
    whiski_chunk_sz=100, n_whiski_chunks=3,
    n_trace_processes=4, expectedrows=1000000, flush_interval=100000):
    # Name the chunks
    chunk_names = ['chunk%04d.tiff' % n_chunk 
        for n_chunk in range(n_whiski_chunks)]

    # read everything
    # need to be able to specify start and stop chunks here
    # need to be able to crop here
    print "Reading"
    frames = my.video.process_chunks_of_video(input_vfile, 
        n_frames=whiski_chunk_sz * n_whiski_chunks,
        func='keep', verbose=True, finalize='listcomp')

    # Dump frames into tiffs or lossless
    print "Writing"
    for n_whiski_chunk, chunk_name in enumerate(chunk_names):
        print n_whiski_chunk
        chunkstart = n_whiski_chunk * whiski_chunk_sz
        chunkstop = (n_whiski_chunk + 1) * whiski_chunk_sz
        chunk = frames[chunkstart:chunkstop]
        write_chunk(chunk, chunk_name)
    
    # Also write lossless and/or lossy monitor video here?
    # would really only be useful if cropping applied

    # trace each
    print "Tracing"
    #~ for n_whiski_chunk, chunk_name in enumerate(chunk_names):
        #~ print chunk_name
        #~ trace_chunk(chunk_name)
    #~ pool = multiprocessing.Pool(n_trace_processes)
    #~ pool.map(trace_chunk, chunk_names)

    # stitch
    print "Stitching"
    setup_hdf5(h5_filename, expectedrows)
    for n_chunk, chunk_name in enumerate(chunk_names):
        append_whiskers_to_hdf5(chunk_name + '.whiskers', 
            h5_filename, chunk_label=n_chunk, verbose=True,
            flush_interval=flush_interval)
