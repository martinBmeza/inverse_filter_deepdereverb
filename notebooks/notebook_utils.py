import glob 
import soundfile as sf
import matplotlib.pyplot as plt

def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    '''List all audio files in a given path and sub-paths. Search for 
    audio formats {wav, flac}. Can be extended to work with MP3 and
    AFF if needed. 
    Parameters: 
        path: string | path where to search audiofiles
        file_types: tuple | tuple containing audio extensions to search
    Returns:
        audio_list: list | list containing the absolute paths of the
        audio files.'''
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list

def dataset_basic_info(file_list):
    samplerates = set()
    durations = set()
    n_channels = set()
    formats = set()
    
    for file in file_list: 
        file_info = sf.info(file)
        samplerates.add(file_info.samplerate)
        durations.add(file_info.duration)
        n_channels.add(file_info.channels)
        formats.add(file_info.format)
        
    print('Sample-rates: {}'.format(samplerates))
    print('Durations in seconds: {}'.format(durations))
    print('Number of channels: {}'.format(n_channels))
    print('Audio formats: {}'.format(formats))



