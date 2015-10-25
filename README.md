# WhiskiWrap
WhiskiWrap provides tools for running whiski (http://whiskertracking.janelia.org) more easily and efficiently. 

My goal is to improve whiski in the following ways:

1. Make it more flexible about reading various input files. In my experience whiski has trouble reading certain input videos. Instead, WhiskiWrap uses your system's ffmpeg to read input files (because ffmpeg can typically read almost anything) and to generate simple tiff stacks which whiski can reliably read.
2. Make it faster, by calling many instances of `trace` in parallel on non-overlapping chunks of the input video.
3. Make it more cross-platform and memory-efficient, by converting whiski's output files into HDF5 files which can be read by multiple programs (Python, Matlab) on any operating system. Importantly, HDF5 files can also be read partially to avoid overflowing your system's memory.


In brief, here is how WhiskiWrap works.

1. Split the entire video into _epochs_ of about 100K frames (~100MB of data). The entire epoch will be read into memory, so the epoch size cannot be too big.
2. For each epoch:
  1. Split it into _chunks_ of about 1000 frames, each of which will be traced separately. The frames can optionally be cropped at this point.
  2. Write each chunk to disk as a tiff stack (note: these files are quite large).
  3. Trace each chunk with parallel instances of `trace`. A `whiskers` file is generated for each chunk.
  4. Parse in order each chunk's `whiskers` file and append the results to an output HDF5 file.
  5. (Optional) delete the intermediate chunk files here.

The following parameters must be chosen:
* `n_trace_processes` - the number of parallel instances of `trace` to run at the same time. The most efficient choice is the number of CPUs on your system.
* `epoch_size` - the number of frames per epoch. It is most efficient to make this value as large as possible. However, it should not be so large that you run out of memory when reading in the entire epoch of video. 100000 is a reasonable choice.
* `chunk_size` - the size of each chunk. Ideally, this should be `epoch_size` / `n_trace_processes`, so that all the processes complete at about the same time. It could also be `epoch_size` / (N * `n_trace_processes`) where N is an integer.

# Installation
WhiskiWrap is written in Python and relies on `ffmpeg` for reading input videos, `tifffile` for writing tiff stacks, `whiski` for tracing whiskers in the tiff stacks, and `pytables` for creating HDF5 files with all of the results.

## Installing `ffmpeg`
First install [`ffmpeg`](https://www.ffmpeg.org/) and ensure it is available on your system path -- that is, you should be able to type `ffmpeg` in the terminal and it should find it and run it.

## Installing `whiski`
Next install [`whiski`](http://whiskertracking.janelia.org). There are several ways to do this:

1. Download the pre-built binary. This is the easiest path because it doesn't require compiling anything. However, you still need to make a few changes to the Python code that is downloaded in order to make it work with `WhiskiWrap`.
2. Build `whiski` from source, using my lightly customized fork. This will probably require more trouble-shooting to make sure all of its parts are working.

To use the pre-built binary:

1. Download the [zipped binary](http://whiskertracking.janelia.org/wiki/display/MyersLab/Whisker+Tracking+Downloads) and unpack it. Rename the unpacked directory to `~/dev/whisk`
2. Add the binaries to your system path so that you can run `trace` from the command line.
3. Add a few files to make `whiski`'s Python code work more nicely with other packages. (Technically, we need to make it a module, and avoid name collisions with the unrelated built-in module `trace`.)
4. `touch ~/whisk/share/whisk/__init__.py`
5. `touch ~/whisk/share/whisk/python/__init__.py`
6. Add these modules to your Python path.
7. `echo "~/whisk/share" >> "~/.local/lib/python2.7/site-packages/whiski_wrap.pth`
8. Test that everything worked by opening python or ipython and running `from whisk.python import traj, trace`

To build from source:

1. Install required dependencies (gl.h, etc)
2. Download the source from my lightly modified fork, which makes the `__init__` changes described above.
3. `cd ~/dev`
4. `git clone https://github.com/cxrodgers/whisk.git`
5. `cd whisk`
6. `mkdir build`
7. `cmake ..`
8. `make`
9. Copy a library into an expected location:
10. `cp ~/dev/whisk/build/libwhisk.so ~/dev/whisk/python`
11. Test that everything worked by opening python or ipython and running `from whisk.python import traj, trace`

## Installing Python modules
Here I outline the use of `conda` to manage and install Python modules. In the long run this is the easiest way. Unfortunately it doesn't work well with user-level `pip`. Specifically, you should not have anything on your `$PYTHONPATH`, and there shouldn't be any installed modules in your `~/.local`.

1. Create a new conda environment for WhiskiWrap.

`conda create -n whiski_wrap python=2.7 pip numpy matplotlib pyqt tables pandas ipython`
2. Activate that environment and install `tifffile`
```
source activate whiski_wrap
pip install tifffile
```
3. Clone WhiskiWrap
```
cd ~/dev
git clone https://github.com/cxrodgers/WhiskiWrap.git
```
4. Make sure the development directory is on your Python path.

`echo "~/dev" >> ~/.local/lib/python2.7/site-packages/whiski_wrap.pth`



