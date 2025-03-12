## whisper.cpp
This is a automatic speech recognition (ASR) library for C/C++.

### Spectrogram
When we process raw audio it is in a wave form which is in the time domain. We
see a single amplitude at each point in time. This amplitude represents the 
total sum of all frequencies at that point in time.

![image](./images/waveform.png)

The wave form is a continuous signal in time and in in amplitude. To represent
this information in a digital form we need to sample it, that is read specific
points (or rather intervals) and store them. These intervalls are ofter denoted
by T, the time between each sample. The number of samples per second is called
the sample rate. The sample rate is often 44.1 kHz or 44,100 samples per second.
Now, if we sample with a low rate we run the risk of losing information so I
though that using a higher sample rate would always be better. But this is not
the case. There is a theorem called the Nyquist-Shannon theorem which states
that the sample rate must be at least twice the highest frequency in the signal
to accurately represent it. So if we have a signal with a maximum frequency of
22 kHz we need a sample rate of at least 44.1 kHz. This is why the sample rate
is often 44.1 kHz. The human ear can hear frequencies up to 20 kHz so this is
why the sample rate is often 44.1 kHz.
Speach is often between 80-8000Hz and music 20-20000Hz.

In this recording I'm just saying my name: "Daniel Bevenius"
I can see that the frequency is not higher than 8000hz which is the limit for
human speech. I start with a frequency at around 100h, which is my fundamental
frequency F0. This is the base vibration rate of my vocal folds.
The "D" sound doesn't have very high amplitude in the waveform because it's what
is call a "stop consonant" - it involves briefly stopping airflow before
releasing it. They tend to be quieter than vowels.
The parallel horizontal lines above the fundamental frequency (at ~200Hz, ~300Hz,
etc.) - these are the harmonics.


The Fourier Transform decomposes this signal into its constituent frequencies.
The x-axis is time just like in the wave form, but the y-axis is frequency (not
amplitude). So the spectrogram is showing the whole spectrum of frequencies
possible for each point in time. Now, the color of each point, the point at
time/frequence (x, y) intersection represents the amplitude/energy at that point.

![image](./images/spectrogram.png)


So these value represent compressed audio in the frequency domain. Now, to recap
a little, normal sound when we record we are capturing the pressure the air
waves are hitting the microphone. This is sampled thousands of times per second
and each number, sample, is the amount of how much the microphone membrane is
pushed in or pushed out. These would be numbers like 0.2, 0.3, -0.1 etc.
An audio signal would be store something like this:
```console
Original Audio --> Fourier Transform --> Magnitude and Phase

Time Domain                              Frequency Domain
```
The Fourier Transform that says that any signal can be represented as a sum of
sine and cosine waves. And each sine wave has two properties magnitude and
phase. Note that this is magnitude and not amplitude so this is a how far value
differs from zero. The magnitude is always positive (the values above in the
first half of embd are in logarithmic scale so they need to be exponentiated to
get the actual values which we can then see are positive). Imagine the final
audio wave for the word "Hello", it will first have a frequencies for the H and
the transition to the frequencies for e, and it need to know where along then
final audio wave to do this transition, this is what phase is for.

### whisper-cli


### Diarization
There is a parameter named 'diarize' which indicates if speaker identification
or diarization should be performed. This is about who spoke when. The system
will attempt to identify the speaker and assign a label to each speaker. So
the output will have an identifier like "Speaker 0", "Speaker 1", etc.
