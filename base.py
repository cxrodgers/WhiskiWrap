import tifffile
import os
#~ import my
import numpy as np
import subprocess
import multiprocessing
import tables
from whisk.python import trace
import re
import datetime 
import shutil
import errno

# Find the repo directory and the default param files
DIRECTORY = os.path.split(__file__)[0]
PARAMETERS_FILE = os.path.join(DIRECTORY, 'default.parameters')
HALFSPACE_DB_FILE = os.path.join(DIRECTORY, 'halfspace.detectorbank')
LINE_DB_FILE = os.path.join(DIRECTORY, 'line.detectorbank')

class FileNamer(object):
    """Defines the naming convention for whiski-related files.
    
    This can be initialized from a basename, such as:
        fn = FileNamer('~/my_directory/session_name')
    or from an existing video file or whiskers file, such as:
        fn = FileNamer.from_whiskers('~/my_directory/session_name.whiskers')
    In the latter case a warning is issued if no such file exists, or if it
    does not follow the typical naming convention.
    
    Once initialized, this object generates names:
        fn.whiskers
        fn.tiff_stack
        fn.video(type='mp4')
    """
    def __init__(self, basename):
        """Initialize based on full path and filename (without extension)."""
        self.basename = os.path.abspath(os.path.expanduser(basename))
    
    def video(self, typ='tif'):
        return self.basename + '.' + typ
    
    @property
    def tiff_stack(self):
        """Returns the name for the tiff stack"""
        return self.video('tif')
    
    @property
    def whiskers(self):
        """Return the name for the whiskers file"""
        return self.basename + '.whiskers'
    
    @classmethod
    def from_video(self, video_name):
        """Generates FileNamer based on an existing video name"""
        if not os.path.exists(video_name):
            print "warning: nonexistent video %s" % video_name
        basename, ext = os.path.splitext(video_name)
        if ext not in ['.mp4', '.avi', '.mkv', '.tif']:
            print "warning: %s does not appear to be a video file" % video_name
        return FileNamer(basename)

    @classmethod
    def from_whiskers(self, whiskers_file_name):
        """Generates FileNamer based on an existing whiskers file"""
        if not os.path.exists(whiskers_file_name):
            print "warning: nonexistent whiskers file %s" % whiskers_file_name        
        basename, ext = os.path.splitext(whiskers_file_name)
        if ext != '.whiskers':
            raise ValueError("%s is not a whiskers file" % whiskers_file_name)
        return FileNamer(basename)       

    @classmethod
    def from_tiff_stack(self, tiff_stack_filename):
        """Generates FileNamer based on an existing tiff stack"""
        if not os.path.exists(tiff_stack_filename):
            print "warning: nonexistent tiff stack %s" % tiff_stack_filename        
        basename, ext = os.path.splitext(tiff_stack_filename)
        if ext != '.tif':
            raise ValueError("%s is not a *.tif stack" % whiskers_file_name)
        return FileNamer(basename)       

    @property
    def hdf5(self):
        """Return the name for the hdf5 file"""
        return self.basename + '.hdf5'


def probe_command_availability(cmd):
    """Try to run 'cmd' in a subprocess and return availability.
    
    'cmd' should be provided in the format expected by subprocess: a string,
    or a list of strings if multiple arguments.
    
    Raises RuntimeError if the called process crashes (eg, via Ctrl+C)
    
    Returns:
        command_available, stdout, stderr
    
    stdout and stderr will be '' if command was not available.
    """
    # Try to initialize a pipe which will only work if it is available
    command_available = True
    try:
        # If it fails here due to nonexistence of command, pipe is
        # never initialized
        pipe = subprocess.Popen(cmd, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return False, '', ''
    
    # This extract data from the pipe
    # I think this will always work?
    try:
        stdout, stderr = pipe.communicate()
    except:
        raise RuntimeError("process crashed")
    
    # Try to terminate it if it didn't already happen
    # Add some point it seemed this was necessary to restored stdout
    # but this no longer seems to be the case
    try:
        pipe.terminate()
    except OSError:
        pass
    
    return command_available, stdout, stderr

def probe_needed_commands():
    """Test whether we have the commands we need.
    
    ffmpeg
    trace
    """
    ffmpeg_av = probe_command_availability('ffmpeg')
    if not ffmpeg_av[0]:
        raise OSError("'ffmpeg' is not available on the system path")    
    if 'the FFmpeg developers' not in ffmpeg_av[2].split('\n')[0]:
        print "warning: libav ffmpeg appears to be installed"
    
    trace_av = probe_command_availability('trace')
    if not trace_av[0]:
        raise OSError("'trace' is not available on the system path")

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

def trace_chunk(chunk_name):
    print "Starting", chunk_name
    orig_dir = os.getcwd()
    fn = FileNamer.from_tiff_stack(chunk_name)
    run_dir, raw_tiff_stack = os.path.split(fn.tiff_stack)
    whiskers_file = fn.whiskers
    command = ['trace', raw_tiff_stack, whiskers_file]

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
    print "Done", chunk_name
    
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

def process_chunks_of_video(filename, 
    frame_start=None, frame_stop=None, n_frames=None, frame_rate=None,
    frame_func=None, chunk_func=None, 
    image_w=None, image_h=None,
    verbose=False,
    frames_per_chunk=1000, bufsize=10**9,
    pix_fmt='gray', finalize='list'):
    """Read frames from video, apply function, return result
    
    The dataflow is:
    1) Use a pipe to ffmpeg to load chunks of frames_per_chunk frames
    2) Apply frame_func to each frame
    3) Apply chunk_func to the chunk
    4) Append the result of chunk_func to a list
    5) "Finalize" that list and return
    
    If n_frames > # available, returns just the available frames with a
    warning.
    
    filename : file to read
    frame_start, frame_stop, n_frames : frame range to process
        If frame_start is None: defaults to zero
        If frame_stop is None: defaults to frame_start + n_frames
        If frame_stop and n_frames are both None: processes the entire video
    frame_rate : used to convert frame_start etc. to times, as required by
        ffmpeg. If None, it will be inferred from ffprobe
    frame_func : function to apply to each frame
        If None, nothing is applied. This obviously requires a lot of memory.
    chunk_func : function to apply to each chunk
        If None, nothing is applied
    image_w, image_h : width and height of video in pixels
        If None, these are inferred using ffprobe
    verbose : If True, prints out frame number for every chunk
    frames_per_chunk : number of frames to load at once from ffmpeg
    bufsize : sent to subprocess.Popen
    pix_fmt : Sent to ffmpeg.
    finalize : Function applied to final result.
        'concatenate' : apply np.concatenate
        'list' : do nothing. In this case the result will be a list of
            length n_chunks, each element of which is an array of length
            frames_per_chunk
        'listcomp' : uses a list comprehension to collapse over the chunks,
            so the result is a list of length equal to the total number of
            frames processed
    
    This function has been modified from my.video to be optimized for
    processing chunks rather than entire videos.
    
    Returns: result, as described above
    """
    # Get aspect
    if image_w is None or image_h is None:
        image_w, image_h, junk = get_video_params(filename)
    if frame_rate is None:
        frame_rate = get_video_params(filename)[2]
    
    # Frame range defaults
    if frame_start is None:
        frame_start = 0
    if frame_stop is None:
        if n_frames is None:
            frame_stop = np.inf
            n_frames = np.inf
        else:
            frame_stop = n_frames - frame_start
    if n_frames is None:
        n_frames = frame_stop - frame_start
    assert n_frames == frame_stop - frame_start
    if frame_stop < frame_start:
        raise ValueError("frame start cannot be greater than frame stop")
    
    # Set up pix_fmt
    if pix_fmt == 'gray':
        bytes_per_pixel = 1
        reshape_size = (image_h, image_w)
    elif pix_fmt == 'rgb24':
        bytes_per_pixel = 3
        reshape_size = (image_h, image_w, 3)
    else:
        raise ValueError("can't handle pix_fmt:", pix_fmt)
    read_size_per_frame = bytes_per_pixel * image_w * image_h
    
    # ffmpeg requires start time and total time to be in seconds, not frames
    # Add 10% of a frame so that it will round down to the correct frame
    start_frame_time = (frame_start + 0.1) / float(frame_rate)
    total_time = (n_frames + 0.1) / float(frame_rate)
    
    # Create the command
    command = ['ffmpeg', 
        '-ss', '%0.4f' % start_frame_time,
        '-i', filename,
        '-t', '%0.4f' % total_time,
        '-f', 'image2pipe',
        '-pix_fmt', pix_fmt,
        '-vcodec', 'rawvideo', '-']
    
    # To store result
    res_l = []
    frames_read = 0

    # Init the pipe
    # We set stderr to PIPE to keep it from writing to screen
    # Do this outside the try, because errors here won't init the pipe anyway
    pipe = subprocess.Popen(command, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        bufsize=bufsize)

    # Catch any IO errors and restore stdout
    try:
        # Read in chunks
        out_of_frames = False
        while frames_read < n_frames and not out_of_frames:
            if verbose:
                print frames_read
            # Figure out how much to acquire
            if frames_read + frames_per_chunk > n_frames:
                this_chunk = n_frames - frames_read
            else:
                this_chunk = frames_per_chunk
            
            # Read this_chunk, or as much as we can
            raw_image = pipe.stdout.read(read_size_per_frame * this_chunk)
            
            # check if we ran out of frames
            if len(raw_image) < read_size_per_frame * this_chunk:
                print "warning: ran out of frames"
                out_of_frames = True
                this_chunk = len(raw_image) / read_size_per_frame
                assert this_chunk * read_size_per_frame == len(raw_image)
            
            # Process
            flattened_im = np.fromstring(raw_image, dtype='uint8')
            if bytes_per_pixel == 1:
                video = flattened_im.reshape(
                    (this_chunk, image_h, image_w))
            else:
                video = flattened_im.reshape(
                    (this_chunk, image_h, image_w, bytes_per_pixel))
            
            # Apply the frame_func to each frame
            # We make it an array again, but note this can lead to 
            # dtype and shape problems later for some frame_func
            if frame_func is not None:
                chunk_res = np.asarray(map(frame_func, video))
            else:
                chunk_res = video
            
            # Apply chunk_func to each chunk
            if chunk_func is not None:
                chunk_res2 = chunk_func(chunk_res)
            else:
                chunk_res2 = chunk_res
            
            # Store the result
            res_l.append(chunk_res2)
            
            # Update
            frames_read += this_chunk

    except:
        raise

    finally:
        # Restore stdout
        pipe.terminate()

        # Keep the leftover data and the error signal (ffmpeg output)
        stdout, stderr = pipe.communicate()

    if frames_read != n_frames:
        # This usually happens when there's some rounding error in the frame
        # times
        raise ValueError("did not read the correct number of frames")

    # Stick chunks together
    if len(res_l) == 0:
        print "warning: no data found"
        res = np.array([])
    elif finalize == 'concatenate':
        res = np.concatenate(res_l)
    elif finalize == 'listcomp':
        res = np.array([item for sublist in res_l for item in sublist])
    elif finalize == 'list':
        res = res_l
    else:
        print "warning: unknown finalize %r" % finalize
        res = res_l
        
    return res

def get_video_params(video_filename):
    """Returns width, height, frame_rate of video using ffprobe"""
    # Video duration and hence start time
    proc = subprocess.Popen(['ffprobe', video_filename],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = proc.communicate()[0]

    # Check if ffprobe failed, probably on a bad file
    if 'Invalid data found when processing input' in res:
        raise ValueError("Invalid data found by ffprobe in %s" % video_filename)
    
    # Find the video stream
    width_height_l = []
    frame_rate_l = []
    for line in res.split("\n"):
        # Skip lines that aren't stream info
        if not line.strip().startswith("Stream #"):
            continue
        
        # Check that this is a video stream
        comma_split = line.split(',')
        if " Video: " not in comma_split[0]:
            continue
        
        # The third group should contain the size and aspect ratio
        if len(comma_split) < 3:
            raise ValueError("malform video stream string:", line)
        
        # The third group should contain the size and aspect, separated
        # by spaces
        size_and_aspect = comma_split[2].split()        
        if len(size_and_aspect) == 0:
            raise ValueError("malformed size/aspect:", comma_split[2])
        size_string = size_and_aspect[0]
        
        # The size should be two numbers separated by x
        width_height = size_string.split('x')
        if len(width_height) != 2:
            raise ValueError("malformed size string:", size_string)
        
        # Cast to int
        width_height_l.append(map(int, width_height))
    
        # The fourth group in comma_split should be %f fps
        frame_rate_fps = comma_split[4].split()
        if frame_rate_fps[1] != 'fps':
            raise ValueError("malformed frame rate:", frame_rate_fps)
        frame_rate_l.append(float(frame_rate_fps[0]))
    
    if len(width_height_l) > 1:
        print "warning: multiple video streams found, returning first"
    return width_height_l[0][0], width_height_l[0][1], frame_rate_l[0]

def get_video_duration(video_filename, return_as_timedelta=False):
    """Return duration of video using ffprobe"""
    # Video duration and hence start time
    proc = subprocess.Popen(['ffprobe', video_filename],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = proc.communicate()[0]

    # Check if ffprobe failed, probably on a bad file
    if 'Invalid data found when processing input' in res:
        raise ValueError(
            "Invalid data found by ffprobe in %s" % video_filename)

    # Parse out start time
    duration_match = re.search("Duration: (\S+),", res)
    assert duration_match is not None and len(duration_match.groups()) == 1
    video_duration_temp = datetime.datetime.strptime(
        duration_match.groups()[0], '%H:%M:%S.%f')
    video_duration = datetime.timedelta(
        hours=video_duration_temp.hour, 
        minutes=video_duration_temp.minute, 
        seconds=video_duration_temp.second,
        microseconds=video_duration_temp.microsecond)    
    
    if return_as_timedelta:
        return video_duration
    else:
        return video_duration.total_seconds()

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
    probe_needed_commands()
    
    # Figure out where to store temporary data
    input_vfile = os.path.abspath(input_vfile)
    input_dir = os.path.split(input_vfile)[0]    

    # Setup the result file
    setup_hdf5(h5_filename, expectedrows)

    # Figure out how many frames and epochs
    duration = get_video_duration(input_vfile)
    frame_rate = get_video_params(input_vfile)[2]
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
        frames = process_chunks_of_video(input_vfile, 
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
            fn = FileNamer.from_tiff_stack(os.path.join(input_dir, chunk_name))
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
                print "aborting"
                return
        
        # Erase
        os.system('rm -rf %s' % directory)
    os.mkdir(directory)
    
    # Copy the input video into the session directory
    new_video_filename = os.path.join(directory, input_video_filename)
    shutil.copyfile(input_video, new_video_filename)
    
    # Copy the parameter files in
    for filename in [PARAMETERS_FILE, HALFSPACE_DB_FILE, LINE_DB_FILE]:
        raw_filename = os.path.split(filename)[1]
        shutil.copyfile(filename, os.path.join(directory, raw_filename))
    
    return FileNamer.from_video(new_video_filename)
