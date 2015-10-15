import os
import subprocess

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
            raise ValueError("%s is not a *.tif stack" % tiff_stack_filename)
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
