"""Utility functions for processing video

process_chunks_of_video : used in this module to load an input video with
    ffmpeg and dump tiff stacks to disk of each chunk.
"""
import os
import numpy as np
import subprocess
import pandas
import re
import datetime
from my.video import get_video_params

def process_chunks_of_video(filename, 
    frame_start=None, frame_stop=None, n_frames=None, frame_rate=None,
    frame_func=None, chunk_func=None, 
    image_w=None, image_h=None,
    verbose=False,
    frames_per_chunk=1000, bufsize=10**9,
    pix_fmt='gray', finalize='list'):
    """Read frames from video, apply function, return result
    
    This has some advantage over my.video.process_chunks_of_video
    but is out of date and not really being used.
    
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
    # It seems to round up .. if I want start_time to be 0.0 and I set it to
    # 0.003 (1/10th of a frame), then the first frame is skipped.
    start_frame_time = (frame_start - 0.1) / float(frame_rate)
    total_time = (n_frames + 0.2) / float(frame_rate)
    
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
