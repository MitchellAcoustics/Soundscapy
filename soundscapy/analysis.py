from acoustics import Signal

def spectrogram(filepath):
    
    s = Signal.from_wav(filepath)
    return s.spectrogram()

