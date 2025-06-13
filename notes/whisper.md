## whisper.cpp
This is a automatic speech recognition (ASR) library for C/C++.

### Spectrogram
When we process raw audio it is in a wave form which is in the time domain. We
see a single amplitude at each point in time. This amplitude represents the 
total sum of all frequencies at that point in time.

![image](./images/waveform.png)

The wave form is a continuous signal in time and in in amplitude. To represent
this information in a digital form we need to sample it, that is read specific
points (at regular intervals) and store them. The interval are often denoted
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

"a" in "Daniel" and "e" in "Bevenius" are vowels and they have a lot of energy
in the higher frequencies. The "a" sound has a lot of energy at around 800Hz and

Consonants like "n", "l", "b", "v" each have distinctive patterns. By using
the spectrogram it is actually possible to "read" what letters are being spoken
and this is what the ASR system use. But the ARS systems are trained on a lot
of millions of examples and use statistical models to predict what is being
said.

The Fourier Transform decomposes this signal into its constituent frequencies
and shows the frequencies on the x-axis and the amplitude on the y-axis. But we
don't have any time information here. This is where the spectrogram comes in
which give use both time and frequency information.
The x-axis is time just like in the wave form, but the y-axis is frequency (not
amplitude). So the spectrogram is showing the whole spectrum of frequencies
possible for each point in time. Now, the color of each point, the point at
time/frequence (x, y) intersection represents the amplitude/energy at that point.

![image](./images/spectrogram.png)

Whisper uses mel spectrogram which is a spectrogram where the frequencies are
converted to the mel scale. The mel scale is a scale of pitches that are
perceived by humans as being equally spaced.

### Mel Spectrograms
In the spectrogram we saw above the y-axis was the frequency in Hz. But this is
not great for humans because we don't perceive frequencies linearly. We perceive
frequencies logarithmically. So what do we mean by that?  
Well, if we have a sound at 1000Hz and another at 2000Hz the difference between
them is 1000Hz. Now, if we have a sound at 10000Hz and another at 11000Hz the
difference is is also 1000Hz. But we perceive the difference between 1000Hz and
2000Hz as being much greater than the difference between 10000Hz and 11000Hz. So
the mel scale is a scale that is designed to mimic the way humans perceive sound.
The mel scale is a logarithmic scale that is linear below 1000Hz and logarithmic
above 1000Hz.

So to recap a little and get an overview of the process:  
1. We start with the original analog signal which is continuous in time and
   amplitude.

2. Digitalize the signal by sampling it at a certain rate to create a digital
   waveform. This is now digital (descrete) but still just amplitude values over
   time.
 
3. Standard spectrogram: The digitized audio undergoes a Short-Time Fourier
   Transform (STFT) to break it into frequency components. Say we sample at a
   rate of 16Khz this would show frequencies up to 8Khz (Nyquist limit).
   A standard spectrogram might have frequency resolution of 10Hz per bin,
   resulting in 800 bins for 8000Hz.

4. Mel spectrogram: This is where the data reduction happens. Instead of keeping
   all the 800 frequency bins, we combine them into just 80 mel scaled bins.
   In this process of converting from the spectrogram to the mel spectrogram we
   apply a set of overlapping triangular filters to the spectrogram which are
   called mel filterbanks. These filters combine multiple frequency bins into
   a single mel bin.

   For example, to create the 10th mel bin, we might take a weighted average of
   linear frequency bins 50-70, giving us a single value that represents that
   entire frequency range.
   After this process, each time frame now has only 80 amplitude values instead
   of 800.

Visualize this as a matrix:
```
spectrogram:
              800 (frequency bins)
        +-----------------------------+
        |                             |
        |                             |
        |                             |  500 (time frames)
        |                             |
        |                             |
        |                             |
        |                             |
        +-----------------------------+
                       |
                       ↓ 
mel spectrogram:
              80 (frequency bins)
        +------------+
        |            |
        |            |
        |            |  500 (time frames)
        |            |
        |            |
        |            |
        |            |
        +------------+
```
This is like resizing an image from 800x500 to 80x500. Like we are compressing
the data and it will take up less space. But also it will contain more of the
information that is important to humans speech which is what we want for 
whisper.

For example, in our above spectrogram we had 8000Hz as the max frequency. And
these are evenly divided into ranges, lets say there might be 800+ frequency
ranges (or bins) for 8000Hz.

This might get divided into 80 "bins/buckets" in the mel spectrogram:
```
Bin 1-20:     0-1000Hz (about 50Hz per bin)
Bin 21-40: 1000-2500Hz (about 75Hz per bin)
Bin 41-60: 2500-5000Hz (about 125Hz per bin)
Bin 61-80: 5000-8000Hz (about 150Hz per bin)
```

Converting to mel spectrograms reduces the input data dimensionality while
preserving perceptually important information.

In whisper.cpp the mel filters are part of the model and are loaded when the
model is loaded:
```c++
    // load mel filters
    {
        auto & filters = wctx.model.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fft);

        filters.data.resize(filters.n_mel * filters.n_fft);
        loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
        BYTESWAP_FILTERS(filters);
    }

```
```console
(gdb) ptype  filters
type = struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;
    std::vector<float> data;
} &
(gdb) p filters.n_mel
$1 = 80
(gdb) p filters.n_fft
$3 = 201
```

The function that will create the mel spectrogram is the following function:
```c++
static bool log_mel_spectrogram(
              whisper_state & wstate,
              const float * samples,
              const int   n_samples,
              const int   /*sample_rate*/,
              const int   frame_size,
              const int   frame_step,
              const int   n_mel,
              const int   n_threads,
              const whisper_filters & filters,
              const bool   debug,
              whisper_mel & mel) {
```
Notice that this takes in not only mel parameters but also STFT parameters so
it performs the windowing, FFT.

```c++
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30
```

Lets take a closer look at the padding:
```c++
    // Calculate the length of padding
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;
```
```console
(gdb) p 16000  * 30
$8 = 480000
(gdb) p frame_size
$9 = 400
(gdb) p frame_size / 2
$10 = 200
```
Now, whisper.cpp always expects 30 seconds of audio but if there is not enough
samples then we will pad with zeros.

For example, lets say we have 5 seconds of audio samples only, that will give
us 5*16000=80000 samples. But for 30 seconds we need 480000 samples. So we need
to 25 seconds more of padding, 25*16000=400000 samples.
```c++
    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
```
In my case I have an audio clip that is close to 30 seconds long:
```console
(gdb) p n_samples
$43 = 472373
(gdb) p n_samples / 16000
$24 = 29
```
```
n_samples = 472373
samples:
    [0                           472373] (~29 seconds)
```

And notice that the above is doing:
```console
(gdb) p n_samples + stage_1_pad + stage_2_pad * 2
$15 = 952773
```
So samples_padded will be:
```console
(gdb) p samples_padded.size()
$21 = 952773
(gdb) p samples_padded.size() / 16000
$25 = 59

samples_padded:
    [0                                                             952773] (~59 seconds)
```
So the number of samples will actually be 59 second worth of samples (952773/16000).
I was not expecting this large number of samples but lets continue and see how
this is used.

Next, the code will copy samples from the original input audio samples passed
to this function. And this will copy the number of samples in the original audio
and the destination is the padded samples we resized above, and starting at
position 200 in this case:
```c++
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);
```
```
samples:
    [0                           472373] (~29 seconds)

samples_padded:
    [0  [0                            472373]                      952773] (~59 seconds)
        200
```
```console
(gdb) p samples_padded[200]
$31 = 0
(gdb) p samples_padded[201]
$32 = -5.04691343e-06
(gdb) p samples[0]
$34 = 0
(gdb) p samples[1]
$35 = -5.04691343e-06
```
So at this point the first 200 values are 0:
```console
(gdb) p samples_padded
$30 = std::vector of length 952773, capacity 952773 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...}
```
Following that we have:
```c++
    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);
```
```console
(gdb) p n_samples
$38 = 472373
(gdb) p n_samples + stage_2_pad
$36 = 472573
```
So the above will fill starting at index 472573, and until
```console
(gdb) p n_samples + stage_1_pad + 2 * stage_2_pad
$42 = 952773

samples_padded:
                                             [ fill                      ]
    [0  [0                            472373]                      952773] (~59 seconds)
        200
```
Then there is a reverse_copy:
```c++
    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());
```
So this is copying from samples[1], and until samples[1 + 200] and then reversing
those values (the reflective part) and copying them to the start of the samples_padded
which currently have 200 zero values at the start:
```console
(gdb) p samples_padded
$51 = std::vector of length 952773, capacity 952773 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...}
(gdb) n
(gdb) p samples_padded
$52 = std::vector of length 952773, capacity 952773 = {1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 1.40129846e-45, 
  -5.60519386e-45, 1.54142831e-44, -4.06376555e-44, 1.0930128e-43, -3.16693453e-43, 7.97338826e-43, 
  -1.98984382e-42, 5.09091732e-42, -1.39541301e-41, 3.53365434e-41, -8.5369905e-41, 2.01459075e-40, 
  -4.87157207e-40, 1.24158407e-39, -2.84790091e-39, 5.67522935e-39, -9.22885079e-39, 2.14333224e-38, 
  -3.76835389e-38, -1.22132242e-38, 4.68280674e-37, -1.54753555e-36, 4.68867058e-36, -1.80823145e-35, 
  7.42930682e-35, -2.25365387e-34, 5.9967477e-34, -1.82701283e-33, 6.2060114e-33, -1.86423853e-32, 4.79754274e-32, 
  -1.34586286e-31, 4.1677613e-31, -1.24611532e-30, 3.15825352e-30, -8.44022473e-30, 2.44651438e-29, 
  -7.24007682e-29, 1.82109508e-28, -4.69445905e-28, 1.28343105e-27, -3.71012531e-27, 9.30824899e-27, 
  -2.32418206e-26, 5.971539e-26, -1.6410385e-25, 4.12798808e-25, -9.95140538e-25, 2.35140069e-24, -5.72890195e-24, 
  1.45563249e-23, -3.31214674e-23, 6.5481501e-23, -1.05963028e-22, 2.56231968e-22, -4.3344042e-22, -2.13231652e-22, 
  5.75005879e-21, -1.76207339e-20, 5.49501639e-20, -2.15708438e-19, 8.89835718e-19, -2.60325692e-18, 
  7.00126167e-18, -2.15786491e-17, 7.38602208e-17, -2.16164767e-16, 5.59511954e-16, -1.58274946e-15, 
  4.94099221e-15, -1.44833367e-14, 3.68185368e-14, -9.89803535e-14, 2.89107524e-13, -8.43319988e-13, 
  2.12309293e-12, -5.4933124e-12, 1.5118462e-11, -4.3325378e-11, 1.08572776e-10, -2.71449141e-10, 7.00887681e-10, 
  -1.92357241e-09, 4.82093743e-09, -1.1599897e-08, 2.7456398e-08, -6.75029384e-08, 1.70501039e-07, -3.85577721e-07, 
  8.01938313e-07, -1.41239252e-06, -2.8569304e-05, -2.9722667e-05, -3.32603413e-05, -6.55875101e-06, 
  1.29710577e-06, -3.1035408e-05, -1.39691601e-05, 1.90523792e-07, -5.40069323e-07, 1.62006461e-06, 
  -2.70863638e-05, -2.83323207e-05, -9.20175012e-07, -3.10223877e-05, -1.44252172e-05, 3.81352902e-06, 
  -2.30503829e-05, -2.98544546e-05, -3.27684393e-05, -5.75928107e-06, 1.04856463e-06, -3.23147142e-05, 
  -1.13924916e-05, -3.28964416e-06, -2.97789829e-05, 1.4207933e-06, -1.8122395e-05, -3.04698606e-05, 
  -2.96604467e-05, -6.34061507e-05, -1.51871718e-05, 1.31870274e-05, -2.36607502e-05, -6.22213265e-05, 
  -4.41496413e-05, -5.04691343e-06...}

samples:
    [[0  200]                 472373] (~29 seconds)
           |
           +----+
samples_padded: |
     +----------+
     ↓                            
    [0  [0                            472373]                      952773] (~59 seconds)
        200
```
So that is the padding done, and next thing to happen is the actual log mel
computation:
```c++
    mel.n_mel     = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);
```
```console
$61 = (whisper_mel &) @0x555556328a88: {n_len = 5952, n_len_org = 2952, n_mel = 80,
  data = std::vector of length 476160, capacity 476160 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
```

```c++
    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded),
                    n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                    std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }
```
Notice that `std::cref` is used for both the padded samples and the mel filters
as these are quite large and we don't want to copy them and we instead pass them
as const references. In a similar way the mel object is passed by ref but it
is mutable. And the created threads will start directly and then the main
thread will also run the same function. And the worker threads will all join
when they are done. Notice that the main thread will pass in 0 as ith.

```c++
static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters, whisper_mel & mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);
```

```console
(gdb) p fft_in.size()
$4 = 800
(gdb) p fft_out.size()
$5 = 3200
```
```c++
    int n_fft = filters.n_fft;
    int i = ith;
```
Notice that `i` is set to the thread number so each thread will process parts
of the padded samples using the frame_step to get the offset into the samples
that the thread is operating on:
```c++
    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;
```
In my case I've set `--threads 1` to simlify debugging so this will always be
zero.

First the Hann window is applied:
```c++
        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
```
Where `hann` is passed in and is in a global precomputed cache:
```c++
    const float * hann = global_cache.hann_window;
```
And the above loop will apply the Hann window to the padded samples either one
frame at a time or all the samples if there are less than the frame size. So
this is doing to make our frame bell shaped starting at zero and then going up
to 1 and then back down to zero which enables us to the stft.
So we are first doing this for samples_padded[0] -> samples_padded[399] and then
for samples_padded[160] -> samples_padded[559] and so on. This is part of the
overlapping frames to avoid missing any information due to the start and end of
the window.
If multiple threads were used it could be possible that the last thread would
have less samples than a frame to process which is the reason for the additional
check.

In a similar manner we also pad/fill with zeros if the number of samples is less
that a frame size:
```c++
        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }
```
Then we do the Fast Fourier Transform:
```c++
        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());
```
After that the output of the fft will be complex number in pairs in `fft_out`.
Where the first is the cosine component, that is how much does this frequency
(this bin) align with the cosine wave for this sample.

We can view these pairs like this:
```
Rectagular form:
z = a + bi

Polar form:
z = r * e^(i * theta)

r     = sqrt(a^2 + b^2)  (magnitude)
theta = atan(b/a)        (phase)
```
The magnitude represents the amplitude of this frequency component.
And what we are doing below is calculating the amplitude of each frequency
component (pair).

If we think of this as going round a circle (the frequency) and then viewing
this from the side (sine) and from below (cosine) this will trace out a curve,
and in this case we are dealing with a point on this circle which maps to a
point in the complex plane. We can think of this as a 2d point. When we calculate
the magnitude we are simply calculating the distance from the origin to this
point using the pythagorean theorem.

So magnitude is the strength/amplitude of the signal at this frequency. But we
can also calulate the power:
```
r     = sqrt(a^2 + b^2)  (magnitude) (how loud/strong this frequency component is)
r²    = a^2 + b^2        (power) (how much energy this frequency component contributes to the total signal)
```
So the magnitude tells us this frequency is present at X amplitude, and power
tell us that this frequency contributes to X% or the total energy of the signal.

Think of a signal with two components:
```
100Hz : magnitude = 2, power = 4
200Hz : magnitude = 4, power = 16

Total power = 20
100Hz contributes 20% of the total power
200Hz contributes 80% of the total power
```
Now, just becuase we are using power here does not mean we loose amplitude, we
can always calculate by taking the square root of the power. For speech analysis
and recognition we are interested in the power spectrum, not the amplitude.
This is also convenient because simply adding the power of each frequency gives
us the tolal energy of the signal. And taking the log of power directly gives
us dB scale.

```c++
        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }
```
```
             [           a^2                        ]   [               b^2                   ]
fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
```
What we get out of this is a power spectrum which shows us how much energy is
distributed accross different frequencies.

Next, we have the mel spectrogram calculation for each mel band which is our
case is 80. So for each mel band will calculate a dot product between the power
in fft_out with the filters. So each power will be multiplied by each filter
and then summed up to become the mel band.
```
Mel band j = Σ(power[k] × filter_j[k]) for all k
```
And filters are triangular and constructed so that the lower powers filter are
closer to each other and then get spread out more as higher powers.
So if we have a low power for power[k] then this would be multiplied by the
filter that covers that interval and this will map it to a specific mel band.

```c++
        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
```
Now, the inner loop does the following:
```
(gdb) p n_fft
$63 = 201

(gdb) p n_fft - 3
$64 = 198
```
So the loop will iterator over 198 values, and it will do so incrementing k
by 4 each time, processing 4 values on each iteration. And since it processes
4 values at a time we need to stop at 198 and not 201.
And the last loop simple handles those values if there are any left over, that
don't fit into groups of four.

And the final part of this function is:
```c++
    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
```
This is handling the case where the number of mel bands is larger than the number
of processable frames. In this case the mel bands are filled with the log of
1e-10. This is a very small value and will be ignored in the final output and
is like silence or background noice floor.

This will then return us to `log_mel_spectrogram` where we will find the 
maximum value accross the entire mel spectrogram:
```c++
    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }
```
Next we will use the max value to normalize the mel spectrogram:
```c++
    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }
```
Notice that anyhing above the max value is set to the max value minus 8.0
(clamping). And then we normalize and these particular values are probably from
the whisper models training. This is something to keep in mind if we want to
start supporting other models.

### Inference Processing
The raw audio is first split into smaller segment of 30 second chunks. This 30
second limit comes from Whispers training where 30 second segments where used
and its position embeddings are designed for this duration.

Then for each chunk:
The chunk is converted into a mel spectrogram which is then processed by the
encoder.

The decoder starts with initial tokens:
* `<SOT>` (Start of Transcript),
* language token (like `<EN>`),
* task token which will be one of:
  - `<TRANSCRIBE>`
  - `<TRANSLATE>`
* no timestamp token

The decoder generates one token at a time.

For a single 30-second chunk, the result will be the transcribed tokens
corresponding to the speech in that 30-second segment. 

The special tokens (like language markers, timestamp indicators) provide task
control. By intermixing these with the text, Whisper can perform multiple tasks
with a single model, directing it to transcribe, translate, or identify the
language.

In whisper.cpp, these tokens are setup in the `whisper_full_with_state` function:
```c++
int whisper_full_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    ...

    std::vector<whisper_token> prompt_init = { whisper_token_sot(ctx), };
```
And we can inspect the tokes added when translation is enabled using the
command line option `-tr`:
```console
(std::vector<int>) size=3 {
  [0] = 50258
  [1] = 50259
  [2] = 50358
}
(lldb) p ctx->vocab.id_to_token.at(50258)
(std::map<int, std::string>::mapped_type) "[_SOT_]"
(lldb) p ctx->vocab.id_to_token.at(50259)
(std::map<int, std::string>::mapped_type) "[_LANG_en]"
(lldb) p ctx->vocab.id_to_token.at(50358)
(std::map<int, std::string>::mapped_type) "[_TRANSLATE_]"
```
Transcribe is token:
```console
(lldb) p ctx->vocab.id_to_token.at(50359)
(std::map<int, std::string>::mapped_type) "[_TRANSCRIBE_]"
```

In Whisper models (including implementations like whisper.cpp), the task token (<|transcribe|> or <|translate|>) fundamentally changes what the model does with the audio input. These are mutually exclusive paths in the model's processing:

<|transcribe|> instructs the model to output text in the same language as the audio
<|translate|> instructs the model to translate the speech into English

To get both the original transcription and an English translation, you would indeed need to run two separate inference passes over the same audio:

First pass with <|transcribe|> to get the original language transcription
Second pass with <|translate|> to get the English translation

This is a fundamental limitation of how the model was designed and trained. It's not simply a software limitation that could be worked around in the implementation - the model architecture itself expects to perform one task at a time.

### Diarization
There is a parameter named 'diarize' which indicates if speaker identification
or diarization should be performed. This is about who spoke when. The system
will attempt to identify the speaker and assign a label to each speaker. So
the output will have an identifier like "Speaker 0", "Speaker 1", etc.

This requires an audio signal with multiple channels I think: 
```c++
static std::string estimate_diarization_speaker(
    std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1, bool id_only = false) {
    std::string speaker = "";
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
    const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++) {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1*energy1) {
        speaker = "0";
    } else if (energy1 > 1.1*energy0) {
        speaker = "1";
    } else {
        speaker = "?";
    }

    if (!id_only) {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}
```
Notice that this is taking `pcmf32s` where the `s` indicates stereo and that this
not a single vector but two vectors. So there are two channels here. 
And it looks like the speakers need to be on separate channels and cannot speak
at the exact same times.
This is performed after the decoding is done.

### Tiny Diarization
This is performed as part of the decoding, and hence the model need to support
this. There is a special token in the vocabulary for this:
```c++
id token_solm       = 50359; // [TDRZ] used by tinydiarize models to indicate speaker turn
```
I found this name a little confusing at first but I think it stands for
start of llm. My understanding is that it is not used in the original whisper
implementation and has been renamed in tinydiarize to `speaker_turn`:
```console
$ git show 7cfca7e
commit 7cfca7e86b3680526b0a070536111135c49ac008
Author: Akash Mahajan <akashmjn@stanford.edu>
Date:   Thu Mar 30 15:18:28 2023 -0700

    Add `small.en-tdrz` checkpoint and initial support for `speakerturn` in decoding results (#4)

    * tmp commit hacking in spkturn decode support

    * rename sot_lm -> speaker_turn

    * add pretrained small.en-tdrz checkpoint

    * update readme with run info
```

https://github.com/akashmjn/tinydiarize




### Pulse Code Modulation (PCM)
So when we a raw audio signal which is continuous, we sample it at a certain
fixed rate called the sample rate. This is measuring the amplitude of the sound
wave at regular intervals. Each value is then quantied to a fixed number of bits
, the number is determined by the bit depth. This gives us a discrete value.
```
8  bits = 2^8  = 256 levels
16 bits = 2^16 = 65536 levels
24 bits = 2^24 = 16777216 levels
32 bits = 2^32 = 4294967296 levels
```
These quantized values are the codes in Pulse Code Modulation (PCM). Each code
represents the amplitude of the sound wave at that point in time. This data
can then be stored in a file and for a WAV file the header will contain metadata
like the sample rate, bit depth, number of channels, etc.

### Waveform Audio File Format (WAV)
This is a subset of Microsoft's Microsoft’s Resource Interchange File Format (RIFF)
specification for the storage of digital audio. There is no compression involved
in this format.

We can inspect the wav metadata using the `mediainfo` command:
```console
$ hexdump -C -n 500 samples/jfk.wav
00000000  52 49 46 46 46 5f 05 00  57 41 56 45 66 6d 74 20  |RIFFF_..WAVEfmt |
00000010  10 00 00 00 01 00 01 00  80 3e 00 00 00 7d 00 00  |.........>...}..|
00000020  02 00 10 00 4c 49 53 54  1a 00 00 00 49 4e 46 4f  |....LIST....INFO|
00000030  49 53 46 54 0e 00 00 00  4c 61 76 66 35 39 2e 32  |ISFT....Lavf59.2|
00000040  37 2e 31 30 30 00 64 61  74 61 00 5f 05 00 00 00  |7.100.data._....|
00000050  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
```
whisper.cpp uses [miniaudio](https://github.com/mackron/miniaudio) to read wav
files.

### Model

```c++
struct whisper_model {
    e_model type = MODEL_UNKNOWN;

    whisper_hparams hparams;
    whisper_filters filters;

    // encoder.positional_embedding
    struct ggml_tensor * e_pe;

    // encoder.conv1
    struct ggml_tensor * e_conv_1_w;
    struct ggml_tensor * e_conv_1_b;

    // encoder.conv2
    struct ggml_tensor * e_conv_2_w;
    struct ggml_tensor * e_conv_2_b;

    // encoder.ln_post
    struct ggml_tensor * e_ln_w;
    struct ggml_tensor * e_ln_b;

    // decoder.positional_embedding
    struct ggml_tensor * d_pe;

    // decoder.token_embedding
    struct ggml_tensor * d_te;

    // decoder.ln
    struct ggml_tensor * d_ln_w;
    struct ggml_tensor * d_ln_b;

    std::vector<whisper_layer_encoder> layers_encoder;
    std::vector<whisper_layer_decoder> layers_decoder;

    // ggml context that contains all the meta information about the model tensors
    struct ggml_context * ctx = nullptr;

    // the model backend data is read-only and can be shared between processors
    ggml_backend_buffer_t buffer = nullptr;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};
```
In `whisper_model_load` the model tensor are created:
```c++
        // encoder
        {
            model.e_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx);
```
This is the encoders position encoder (pe)
```console
(gdb) p model.e_pe.ne
$35 = {384, 1500, 1, 1}

(gdb) p model.hparams.n_audio_state 
$9 = 384
(gdb) p model.hparams.n_audio_ctx 
$10 = 1500
```
So we can see this is a matrix with 384 dimensions and 1500 rows.
```
0     [0                  383]
            .
            .
            .
1499  [0                  383]
```
Audio is processed at 50 frames per second so 1500 frames corresponds to 30
seconds of audio (1500/50 = 30). So each row in this matrix represents a specific
time position the 30 second audio chunk or segment. The positional information
is added so that the model know not only the spectral information (which frequencies
are present) but also when they occur.

Next we have the two 1D convolutions:
```c++
            model.e_conv_1_w     = ggml_new_tensor_3d(ctx, vtype,         3, n_mels,     n_audio_state);
            model.e_conv_1_b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,         1,     n_audio_state);

            model.e_conv_2_w     = ggml_new_tensor_3d(ctx, vtype,         3, n_audio_state, n_audio_state);
            model.e_conv_2_b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,                1, n_audio_state);
```
```console
(gdb) p model.e_conv_1_w->ne
$12 = {3, 80, 384, 1}

(gdb) p model.e_conv_2_w->ne
$13 = {3, 384, 384, 1}
```
TODO: Add more information about the convolutions.

Next, we have the encoder layers.
And following that we have the decoder tensors.
```c++
        // decoder
        {
            model.d_pe   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_text_state, n_text_ctx);

            model.d_te   = ggml_new_tensor_2d(ctx, wtype,         n_text_state, n_vocab);
```
These are the decoders positional embedding and token embedding tensors.
```console
(gdb) p model.d_pe->ne
$19 = {384, 448, 1, 1}

(gdb) p model.d_te->ne
$20 = {384, 51864, 1, 1}
```
These tensors are later used in :
```c++
static struct ggml_cgraph * whisper_build_graph_conv(
        whisper_context & wctx,
          whisper_state & wstate) {
    ...
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2*n_ctx, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor * cur = nullptr;

    if (!whisper_encode_external(wstate)) {
        // convolution + gelu
        {
            cur = ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1);
            cur = ggml_add(ctx0, cur, model.e_conv_1_b);

            cur = ggml_gelu(ctx0, cur);

            cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
            cur = ggml_add(ctx0, cur, model.e_conv_2_b);

            cur = ggml_gelu(ctx0, cur);
        }

        ggml_set_name(cur, "embd_conv");
        wstate.embd_conv = cur;
```
```console
(gdb) p mel->ne
$31 = {3000, 80, 1, 1}
```

### whisper-cli
An initial walk through of the cli example to get familiar with the code.
```console
gdb --args ./build/bin/whisper-cli \
	-m models/ggml-tiny.en.bin \
	-f samples/jfk.wav \
	-di
```

```c++
int main(int argc, char ** argv) {
    whisper_params params;
    ...

    struct whisper_context_params cparams = whisper_context_default_params();
    ...
    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
}
```
```console
(gdb) set print pretty on
(gdb) p params
$4 = {
  n_threads = 4,
  n_processors = 1,
  offset_t_ms = 0,
  offset_n = 0,
  duration_ms = 0,
  progress_step = 5,
  max_context = -1,
  max_len = 0,
  best_of = 5,
  beam_size = 5,
  audio_ctx = 0,
  word_thold = 0.00999999978,
  entropy_thold = 2.4000001,
  logprob_thold = -1,
  no_speech_thold = 0.600000024,
  grammar_penalty = 100,
  temperature = 0,
  temperature_inc = 0.200000003,
  debug_mode = false,
  translate = false,
  detect_language = false,
  diarize = true,
  tinydiarize = false,
  split_on_word = false,
  no_fallback = false,
  output_txt = false,
  output_vtt = false,
  output_srt = false,
  output_wts = false,
  output_csv = false,
  output_jsn = false,
  output_jsn_full = false,
  output_lrc = false,
  no_prints = false,
  print_special = false,
  print_colors = false,
  print_progress = false,
  no_timestamps = false,
  log_score = false,
  use_gpu = true,
  flash_attn = false,
  suppress_nst = false,
  language = "en",
  prompt = "",
  font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
  model = "models/ggml-tiny.en.bin",
  grammar = "",
  grammar_rule = "",
  tdrz_speaker_turn = " [SPEAKER_TURN]",
  suppress_regex = "",
  openvino_encode_device = "CPU",
  dtw = "",
  fname_inp = std::vector of length 1, capacity 1 = {"samples/jfk.wav"},
  fname_out = std::vector of length 0, capacity 0,
  grammar_parsed = {
    symbol_ids = std::map with 0 elements,
    rules = std::vector of length 0, capacity 0
  }
}

gdb) p cparams
$5 = {
  use_gpu = true,
  flash_attn = false,
  gpu_device = 0,
  dtw_token_timestamps = false,
  dtw_aheads_preset = WHISPER_AHEADS_NONE,
  dtw_n_top = -1,
  dtw_aheads = {
    n_heads = 0,
    heads = 0x0
  },
  dtw_mem_size = 134217728
}

```
So lets take a look at the model is loaded:
```console
(gdb) br whisper_init_from_file_with_params_no_state
(gdb) c
```
```c++
struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model,
    struct whisper_context_params params) {
    WHISPER_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);


```

### Beam Search
Lets take the whisper-cli as the example and see how the beam search works.
```c++
int main(int argc, char ** argv) {
    ...
    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
```
```c++
struct whisper_context * whisper_init_from_file_with_params(const char * path_model, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_from_file_with_params_no_state(path_model, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}
```
```c++
struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model, struct whisper_context_params params) {
    WHISPER_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);
#ifdef _MSC_VER
    // Convert UTF-8 path to wide string (UTF-16) for Windows, resolving character encoding issues.
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring path_model_wide = converter.from_bytes(path_model);
    auto fin = std::ifstream(path_model_wide, std::ios::binary);
#else
    auto fin = std::ifstream(path_model, std::ios::binary);
#endif
    if (!fin) {
        WHISPER_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    whisper_model_loader loader = {};
    ...

    auto ctx = whisper_init_with_params_no_state(&loader, params);
```
```c++
struct whisper_context * whisper_init_with_params_no_state(struct whisper_model_loader * loader, struct whisper_context_params params) {
    ...

    whisper_context * ctx = new whisper_context;
    ctx->params = params;

    if (!whisper_model_load(loader, *ctx)) {
        loader->close(loader->context);
        WHISPER_LOG_ERROR("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }
}
```
```c++
static bool whisper_model_load(struct whisper_model_loader * loader, whisper_context & wctx) {
    WHISPER_LOG_INFO("%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto & model = wctx.model;
    auto & vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            WHISPER_LOG_ERROR("%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }
```
TODO: take a closer look at the verify magic check in combination with using a Core ML
model. This check forces there to be an ggml model even though there might be cases
where only a Core ML model is used.

The actual inference is started by `cli.cpp`:
```c++
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    ...
    if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
        fprintf(stderr, "%s: failed to process audio\n", argv[0]);
        return 10;
    }
```
```c++
int whisper_full_parallel(
        struct whisper_context * ctx,
        struct whisper_full_params params,
        const float * samples,
        int n_samples,
        int n_processors) {
    if (n_processors == 1) {
        return whisper_full(ctx, params, samples, n_samples);
    }
```
```c++
int whisper_full(
        struct whisper_context * ctx,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    return whisper_full_with_state(ctx, ctx->state, params, samples, n_samples);
}
```
```c++
int whisper_full_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    ...
    if (n_samples > 0) {
        // compute log mel spectrogram
        if (whisper_pcm_to_mel_with_state(ctx, state, samples, n_samples, params.n_threads) != 0) {
            WHISPER_LOG_ERROR("%s: failed to compute log mel spectrogram\n", __func__);
            return -2;
        }
    }
}
```
```console
(lldb) p n_samples
(int) 176000
```

Next we have the temperatures:
```c++
    // a set of temperatures to use
    // [ t0, t0 + delta, t0 + 2*delta, ..., < 1.0f + 1e-6f ]
    std::vector<float> temperatures;
    if (params.temperature_inc > 0.0f) {
        for (float t = params.temperature; t < 1.0f + 1e-6f; t += params.temperature_inc) {
            temperatures.push_back(t);
        }
    } else {
        temperatures.push_back(params.temperature);
    }
```
```console
(std::vector<float>) size=6 {
  [0] = 0
  [1] = 0.200000003
  [2] = 0.400000006
  [3] = 0.600000024
  [4] = 0.800000011
  [5] = 1
}
```
These are described in the whisper paper. So we start with a temperature of 0,
and increment in approximately 0.2 steps up to 1.0.
So for a single inference, from the point of view of the caller, it would potentially cause
multiple inferences to happen if the heuristic (issues discovered in practice) happen.
From the caller's perspective, what appears to be a single transcription request could
potentially trigger multiple inference passes behind the scenes.

Next, the decoders are initalized:
```c++
    // initialize the decoders
    int n_decoders = 1;

    switch (params.strategy) {
        case WHISPER_SAMPLING_GREEDY:
            {
                n_decoders = params.greedy.best_of;
            } break;
        case WHISPER_SAMPLING_BEAM_SEARCH:
            {
                n_decoders = std::max(params.greedy.best_of, params.beam_search.beam_size);
            } break;
    };

    n_decoders = std::max(1, n_decoders);

    if (n_decoders > WHISPER_MAX_DECODERS) {
        WHISPER_LOG_ERROR("%s: too many decoders requested (%d), max = %d\n", __func__, n_decoders, WHISPER_MAX_DECODERS);
        return -4;
    }

    // TAGS: WHISPER_DECODER_INIT
    for (int j = 1; j < n_decoders; j++) {
        auto & decoder = state->decoders[j];

        decoder.sequence.tokens.reserve(state->decoders[0].sequence.tokens.capacity());

        decoder.probs.resize   (ctx->vocab.n_vocab);
        decoder.logits.resize  (ctx->vocab.n_vocab);
        decoder.logprobs.resize(ctx->vocab.n_vocab);
        decoder.logits_id.reserve(ctx->model.hparams.n_vocab);

        decoder.rng = std::mt19937(0);
    }
```
Notice that all decoders are initialized with rng seed 0. I opened an issue
about this and it has since been updated.

Then the prompt is prepared:
```c++
    // prepare prompt
    {
        std::vector<whisper_token> prompt_tokens;

        // initial prompt
        if (!params.prompt_tokens && params.initial_prompt) {
            prompt_tokens.resize(1024);
            int n_needed = whisper_tokenize(ctx, params.initial_prompt, prompt_tokens.data(), prompt_tokens.size());
            if (n_needed < 0) {
                prompt_tokens.resize(-n_needed);
                n_needed = whisper_tokenize(ctx, params.initial_prompt, prompt_tokens.data(), prompt_tokens.size());
            }
            prompt_tokens.resize(n_needed);
            params.prompt_tokens   = prompt_tokens.data();
            params.prompt_n_tokens = prompt_tokens.size();
        }

        // prepend the prompt tokens to the prompt_past
        if (params.prompt_tokens && params.prompt_n_tokens > 0) {
            // parse tokens from the pointer
            for (int i = 0; i < params.prompt_n_tokens; i++) {
                prompt_past.push_back(params.prompt_tokens[i]);
            }
            std::rotate(prompt_past.begin(), prompt_past.end() - params.prompt_n_tokens, prompt_past.end());
        }
    }
```
Then the tokens are set up:
```c++
    // these tokens determine the task that will be performed
    std::vector<whisper_token> prompt_init = { whisper_token_sot(ctx), };

    if (whisper_is_multilingual(ctx)) {
        const int lang_id = whisper_lang_id(params.language);
        state->lang_id = lang_id;
        prompt_init.push_back(whisper_token_lang(ctx, lang_id));
        if (params.translate) {
            prompt_init.push_back(whisper_token_translate(ctx));
        } else {
            prompt_init.push_back(whisper_token_transcribe(ctx));
        }
    }
```
`sot` is start of transcript. 

`whisper_state` contains a a list of `whisper_decoder`:
```c++
struct whisper_state {
    ...
    whisper_decoder decoders[WHISPER_MAX_DECODERS];
    ...
};
```
```c++
struct whisper_decoder {
    // the currently generated sequence of tokens
    whisper_sequence sequence;

    // grammar parse state of generated sequence of tokens
    whisper_grammar  grammar;

    int i_batch;    // the index of the token in the current batch
    int seek_delta; // the window shift found so far based on the decoded timestamp tokens

    bool failed;    // has the current segment failed to decode?
    bool completed; // has the decoder completed the current segment?
    bool has_ts;    // have we already sampled a non-beg timestamp token for the current segment?

    // new token probs, logits and logprobs after the last whisper_decode (1-dimensional array: [n_vocab])
    std::vector<float> probs;
    std::vector<float> logits;
    std::vector<float> logprobs;

    // work container used to avoid memory allocations
    std::vector<whisper_pair<double, whisper_vocab::id>> logits_id;

    mutable std::mt19937 rng; // used for sampling at t > 0.0
};

struct whisper_sequence {
    std::vector<whisper_token_data> tokens;

    // the accumulated transcription in the current iteration (used to truncate the tokens array)
    int result_len;

    double sum_logprobs_all; // the sum of the log probabilities of the tokens
    double sum_logprobs;     // the sum of the log probabilities of the tokens (first result_len tokens)
    double avg_logprobs;     // the average log probability of the tokens
    double entropy;          // the entropy of the tokens
    double score;            // likelihood rank score
};
```

```console
-bo N,     --best-of N         [5      ] number of best candidates to keep
-bs N,     --beam-size N       [5      ] beam size for beam search
```

```c++
int whisper_full_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    ...
    // initialize the decoders
    int n_decoders = 1;

    switch (params.strategy) {
        case WHISPER_SAMPLING_GREEDY:
            {
                n_decoders = params.greedy.best_of;
            } break;
        case WHISPER_SAMPLING_BEAM_SEARCH:
            {
                n_decoders = std::max(params.greedy.best_of, params.beam_search.beam_size);
            } break;
    };

    n_decoders = std::max(1, n_decoders);

    if (n_decoders > WHISPER_MAX_DECODERS) {
        WHISPER_LOG_ERROR("%s: too many decoders requested (%d), max = %d\n", __func__, n_decoders, WHISPER_MAX_DECODERS);
        return -4;
    }

    // TAGS: WHISPER_DECODER_INIT
    for (int j = 1; j < n_decoders; j++) {
        auto & decoder = state->decoders[j];

        decoder.sequence.tokens.reserve(state->decoders[0].sequence.tokens.capacity());

        decoder.probs.resize   (ctx->vocab.n_vocab);
        decoder.logits.resize  (ctx->vocab.n_vocab);
        decoder.logprobs.resize(ctx->vocab.n_vocab);
        decoder.logits_id.reserve(ctx->model.hparams.n_vocab);

        decoder.rng = std::mt19937(0);
    }
    ...
    struct beam_candidate {
        int decoder_idx;  // which decoder this candidate came from.
        int seek_delta;   // position in the audio?

        bool has_ts;      // has timestamp information.

        whisper_sequence sequence; // the token sequence for this candidate
        whisper_grammar grammar;   // the grammar for this candidate
    };

    std::vector<std::vector<beam_candidate>> bc_per_dec(n_decoders);
    std::vector<beam_candidate> beam_candidates;
   
    ...

    struct beam_candidate {
        int decoder_idx;
        int seek_delta;

        bool has_ts;

        whisper_sequence sequence;
        whisper_grammar grammar;
    };

    std::vector<std::vector<beam_candidate>> bc_per_dec(n_decoders);
    std::vector<beam_candidate> beam_candidates;

    // main loop
    while (true) {
        if (params.progress_callback) {
            const int progress_cur = (100*(seek - seek_start))/(seek_end - seek_start);

            params.progress_callback(
                ctx, state, progress_cur, params.progress_callback_user_data);
        }

        // if only 1 second left, then stop
        if (seek + 100 >= seek_end) {
            break;
        }

        if (params.encoder_begin_callback) {
            if (params.encoder_begin_callback(ctx, state, params.encoder_begin_callback_user_data) == false) {
                WHISPER_LOG_ERROR("%s: encoder_begin_callback returned false - aborting\n", __func__);
                break;
            }
        }

        // encode audio features starting at offset seek
        if (!whisper_encode_internal(*ctx, *state, seek, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {
            WHISPER_LOG_ERROR("%s: failed to encode\n", __func__);
            return -6;
        }
```
So here we have the inference loop. First the encoder is called which is taking the log mel
spectrogram and passing it to the encoder part of the model. A nice diagram of this can be
found on page 4 of the paper. First there is a Cov1D layer followed by a GELU activation,
and then another Conv1D layer followed by another GELU activation. The output of this then
has position encodings added to it.
This function is also what delegates to the Core ML or OpenVINO external encoders if one
of them are enabled.

```c++
const auto tokens_new = whisper_sample_token_topk(*ctx, decoder, params.beam_search.beam_size);
```
```console
(lldb) p params.beam_search.beam_size
(int) 5

(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```

```c++
// init new transcription with sot, language (opt) and task tokens
prompt.insert(prompt.end(), prompt_init.begin(), prompt_init.end());
```
```console
(lldb) p prompt
(std::vector<int>) size=1 {
  [0] = 50257
}
```
Then we have the decoding part of the model:
```c++
                whisper_kv_cache_clear(state->kv_self);

                whisper_batch_prep_legacy(state->batch, prompt.data(), prompt.size(), 0, 0);

                if (!whisper_decode_internal(*ctx, *state, state->batch, params.n_threads, false, params.abort_callback, params.abort_callback_user_data)) {
                    WHISPER_LOG_ERROR("%s: failed to decode\n", __func__);
                    return -8;
                }
```
```c++
                // Calculate no_speech probability after first decode.
                // This has to be done before any logit filtering. Hence we cannot use the probs from the whisper_process_logits.
                {
                    const int n_logits = ctx->vocab.id_to_token.size();
                    std::vector<float> logprobs(n_logits);
                    std::vector<float> probs(n_logits);

                    whisper_compute_logprobs(state->logits, n_logits, logprobs);
                    whisper_compute_probs(state->logits, n_logits, logprobs, probs);
                    state->no_speech_prob = probs[whisper_token_nosp(ctx)];
                }
```
```c++
                {
                    const int64_t t_start_sample_us = ggml_time_us();

                    state->decoders[0].i_batch = prompt.size() - 1;

                    whisper_process_logits(*ctx, *state, state->decoders[0], params, t_cur);

                    for (int j = 1; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        whisper_kv_cache_seq_cp(state->kv_self, 0, j, -1, -1);

                        memcpy(decoder.probs.data(),    state->decoders[0].probs.data(),    decoder.probs.size()*sizeof(decoder.probs[0]));
                        memcpy(decoder.logits.data(),   state->decoders[0].logits.data(),   decoder.logits.size()*sizeof(decoder.logits[0]));
                        memcpy(decoder.logprobs.data(), state->decoders[0].logprobs.data(), decoder.logprobs.size()*sizeof(decoder.logprobs[0]));
                    }

                    state->t_sample_us += ggml_time_us() - t_start_sample_us;
                }
```

Notice that this is passing the first decoder to the `whisper_process_logits` function:
```c++
static void whisper_process_logits(
              struct whisper_context & ctx,
               struct whisper_state  & state,
              struct whisper_decoder & decoder,
    const struct whisper_full_params   params,
                               float   temperature) {
        ...
        if (params.suppress_blank) {
            if (is_initial) {
                logits[vocab.token_eot]           = -INFINITY;
                logits[vocab.token_to_id.at(" ")] = -INFINITY;
            }
        }
```
And in this case the filtering is basically just setting the vocab tokens for these specific
tokens to negative infinity so they will not have any influence on the logprobability calculation?
In this case the filter is preventing the model from starting a transcription with a blank space
or an end-of-transcript token.
This is also how timestamps are not generated:
```c++
        if (params.no_timestamps) {
            for (int i = vocab.token_beg; i < n_logits; ++i) {
                logits[i] = -INFINITY;
            }
        }
```
When we set a logit which recall is the raw values (unnormalized probabilities) from the
final layer of the decoder. They can range from very negative to very positive values.
So they are not constrained to any range like [0,1] and do not sum to 1. There basically
"confidence scores" that indicate how strongly the model believes that a particular token
should be the next token in the sequence.

When we set a logit to negative infinity, we are effectively saying that the probability
of that token is zero. This is a way to filter out tokens that we do not want to consider
in the decoding process.
```c++
        whisper_compute_logprobs(logits, n_logits, logprobs);
```

```c++
static void whisper_compute_logprobs(
                const std::vector<float> & logits,
                              const int    n_logits,
                      std::vector<float> & logprobs) {
    const float logit_max = *std::max_element(logits.begin(), logits.end());
    float logsumexp = 0.0f;
    for (int i = 0; i < n_logits; ++i) {
        if (logits[i] > -INFINITY) {
            logsumexp += expf(logits[i] - logit_max);
        }
    }
    logsumexp = logf(logsumexp) + logit_max;

    for (int i = 0; i < n_logits; ++i) {
        if (logits[i] > -INFINITY) {
            logprobs[i] = logits[i] - logsumexp;
        } else {
            logprobs[i] = -INFINITY;
        }
    }
}
```
So we first get the maximum logit value. This is used for numerical stability, notice
that is is subtracted from the logits before exponentiation. And also notice that
logits with the value of -INFINITY are not considered in the calculation, which is
what the filtering above is for.

After processing the logits the decoders (starting from 1) are updated:
```c++
                    for (int j = 1; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        whisper_kv_cache_seq_cp(state->kv_self, 0, j, -1, -1);

                        memcpy(decoder.probs.data(),    state->decoders[0].probs.data(),    decoder.probs.size()*sizeof(decoder.probs[0]));
                        memcpy(decoder.logits.data(),   state->decoders[0].logits.data(),   decoder.logits.size()*sizeof(decoder.logits[0]));
                        memcpy(decoder.logprobs.data(), state->decoders[0].logprobs.data(), decoder.logprobs.size()*sizeof(decoder.logprobs[0]));
                    }
```
So this is setting the logits which are the raw values (unnormalized probabilities). And also
the logprobs which are calculated by:
```c++
logprobs[i] = log(exp(logits[i]) / sum(exp(logits)))
```
Logprobs will always be <= 0.0, with 0 representing a probability of 100% probability and
very negative values representing probilities close to 0. These are useful for working with
scoring when computing scores and ranking in beam search.

And the decoders also have the actual normalized probabilites of each token. This is obtained
by applying the softmax function to the logits.

The we have the process function()
```c++

case whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH:
    {
        const auto tokens_new = whisper_sample_token_topk(*ctx, decoder, params.beam_search.beam_size);

        for (const auto & token : tokens_new) {
            bc_per_dec[j].push_back({ j, decoder.seek_delta, decoder.has_ts, decoder.sequence, decoder.grammar, });
            bc_per_dec[j].back().sequence.tokens.push_back(token);
            bc_per_dec[j].back().sequence.sum_logprobs_all += token.plog;
        }
    } break;
```
So for each of the decoders we are going to call `whisper_sample_token_topk`:
```c++
static std::vector<whisper_token_data> whisper_sample_token_topk(
            whisper_context & ctx,
            whisper_decoder & decoder,
                        int   k) {
    ...
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    for (int i = 0; i < k; ++i) {
        const auto id = dist(decoder.rng);
        //printf("XXX %d %d %f %f %f %f\n", id, tid, probs[id], logprobs[id], pt, ptsum);

        result.push_back({ id, tid, probs[id], logprobs[id], pt, ptsum, -1, -1, -1, 0.0f, });

        if (result[i].id >= vocab.token_beg) {
            result[i].tid = result[i].id;
            result[i].pt  = result[i].p;
        }
    }
```
Notice that this is using dist which is a discrete distribution. Each decoder has its own
random number generator so when sampling from this distribution, which is done k times,
the values will be different.
```c++
    std::discrete_distribution<> dist(probs.begin(), probs.end());
```
This creates a distribution where the probability of selecting token i is
proporitional to probs[i], the probability value. Higher probability values are more
likely and lower probability values are less likely.

```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
```
Now, recall that each sample is independent when sampled from the distribution and that the
token 50363 has a probability of 0.837067842. So it is not surprising that it is sampled
multiple times.

These are the values for j=1:
```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```
These are the values for j=2:
```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```
These are the values for j=3:
```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```
These are the values for j=4:
```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```

### Alignment Heads
There is a struct name `whisper_ahead` which stands for alignment heads.
This struct is used with dynamic time warping (DTW) to align the tokens with
the audio frames.

So different models will often have different numbers of attention heads and
this means they might have different predefined sets of alignment heads too.
```c++
    enum whisper_alignment_heads_preset {
        WHISPER_AHEADS_NONE,
        WHISPER_AHEADS_N_TOP_MOST,  // All heads from the N-top-most text-layers
        WHISPER_AHEADS_CUSTOM,
        WHISPER_AHEADS_TINY_EN,
        WHISPER_AHEADS_TINY,
        WHISPER_AHEADS_BASE_EN,
        WHISPER_AHEADS_BASE,
        WHISPER_AHEADS_SMALL_EN,
        WHISPER_AHEADS_SMALL,
        WHISPER_AHEADS_MEDIUM_EN,
        WHISPER_AHEADS_MEDIUM,
        WHISPER_AHEADS_LARGE_V1,
        WHISPER_AHEADS_LARGE_V2,
        WHISPER_AHEADS_LARGE_V3,
        WHISPER_AHEADS_LARGE_V3_TURBO,
    };
```
So let take a closer look at `WHISPER_AHEADS_BASE_EN`
```c++
static const std::map<whisper_alignment_heads_preset, whisper_aheads> g_aheads {
    ...
    { WHISPER_AHEADS_BASE_EN,   {  5, g_aheads_base_en   } },
    ...
};
```
```c++
static const whisper_ahead g_aheads_base_en[]   = { {3, 3}, {4, 7}, {5, 1}, {5, 5}, {5, 7} };
```
So this is saying that the `base.en` model has 5 alignment heads where the first
number in each pair is the `transformer layer` and the second number is the head
within that layer. For example
```c++
{4, 7}
4 = transformer layer
7 = head number
```

So DTW is used to capture token-level timestamps, which is like figuring out
when each word was spoken in the audio.

This is then included in the whisper context params:
```c++

    struct whisper_context_params {
        bool  use_gpu;
        bool  flash_attn;
        int   gpu_device;  // CUDA device

        // [EXPERIMENTAL] Token-level timestamps with DTW
        bool dtw_token_timestamps;
        enum whisper_alignment_heads_preset dtw_aheads_preset;

        int dtw_n_top;
        struct whisper_aheads dtw_aheads;

        size_t dtw_mem_size; // TODO: remove
    };
```
The defaults look like this:
```c++
struct whisper_context_params whisper_context_default_params() {
    struct whisper_context_params result = {
        /*.use_gpu              =*/ true,
        /*.flash_attn           =*/ false,
        /*.gpu_device           =*/ 0,

        /*.dtw_token_timestamps =*/ false,
        /*.dtw_aheads_preset    =*/ WHISPER_AHEADS_NONE,
        /*.dtw_n_top            =*/ -1,
        /*.dtw_aheads           =*/ {
            /*.n_heads          =*/ 0,
            /*.heads            =*/ NULL,
        },
        /*.dtw_mem_size         =*/ 1024*1024*128,
    };
    return result;
}
```

### KV Caches in whisper.cpp
In whisper.cpp there are three kv-caches. One for the self-attention, one for
the cross attention, and one for padding (flash attention I think).

Now, the self-attention cache is what we might be used to from a normal LLM
where the key and values are computed for each generated token, and then stored
in the cache to save computating them again.

The encoder processes the audio frames and generates some form of embeddings for
them. We can think of then as a matrix where each row represents a certain time
interval in the input audio. This matrix is what is used to populate the cross
attention cache and it does not change during the decoding process, it is of
fixed size and content. This matrix, or sequence of vector embeddings each vector
corresponds to about 20ms of audio.

- Each row represents a specific time interval in the original audio
- The number of columns equals the embedding dimension (the model's hidden state size)

Also recall that there can be multiple decoders in whisper but there is still
only one kv-cache in the state. Each decoder is like a separate sequence and can
use a separate sequence id for its entries in the cache.

So the first cache to be updated is the cross kv cache as the encoder is first.
That is then fixed for the duration of the decoding. After that the decoding
will proceed and it will start with a single token (start of sequence) which
till then attend to the cross attention and after that a self attention. And
the result of the Key and Value for this will be appended to the self attention
cache.

So in the decoder transformer blocks, each block will first have self-attention
then cross-attention, followed by a feed forward network (MLP).

```c++
struct whisper_state * whisper_init_state(whisper_context * ctx) {
    ...
    state->kv_self_n_dec = 1;
    if (!whisper_kv_cache_init(state->kv_self, state->backends[0], ctx->itype,
                ctx->model.hparams.n_text_state,
                ctx->model.hparams.n_text_layer,
                GGML_PAD(ctx->model.hparams.n_text_ctx, 256))) {
        WHISPER_LOG_ERROR("%s: whisper_kv_cache_init() failed for self-attention cache\n", __func__);
        whisper_free_state(state);
        return nullptr;
    }
```
And this is what the signature of the `whisper_kv_cache_init` function looks
like:
```c++
static bool whisper_kv_cache_init(
             struct whisper_kv_cache & cache,
                      ggml_backend_t   backend,
                           ggml_type   wtype,
                             int64_t   n_text_state,
                             int64_t   n_text_layer,
                                 int   n_ctx) {
    const int64_t n_mem      = n_text_layer*n_ctx;
    const int64_t n_elements = n_text_state*n_mem;
```
```console
(gdb) p ctx->model.hparams.n_text_state
$8 = 512
(gdb) p ctx->model.hparams.n_text_layer
$9 = 6
(gdb) p ctx->model.hparams.n_text_ctx
$10 = 448
(gdb) p ctx->itype
$11 = GGML_TYPE_F16
(gdb) p n_ctx
$30 = 512
```
Recall that this particular cache if for the self-attention, that is the decoder.
So in this case we have 6 layers. And each layer need to store/cache up to
c_ctx (512) tokens.
```
Key cache:
           <- n_text_state   ->
Layer 0: 0 [0              512]   ^                  Per layer: 512 * 512 = 262144
                  ...             |
                  ...             n_ctx
                  ...             |
       511 [0              512]   v
...
...

Layer 5: 0 [0              512]   ^
                  ...             |
                  ...             n_ctx
                  ...             |
       511 [0              512]   v

Total: 6 * 262144 = 1572864 (values of some type like float16 etc)

And we have the same for the Value cache
Total: 6 * 262144 = 1572864

Key + Value = 3145728

And if we have 2 bytes per element/value:
3145728 * 2 = 6291456 bytes (6MB)
```
So we have a max context size of 512 tokens. And we have 6 layers. This means
that there will be 6 self-attention "blocks" in the decoder and each one will
have to store the computation of the Key and Values for that block.

```console
(gdb) p n_mem
$15 = 3072
(gdb) p n_elements
$16 = 1572864
```
If we look at the `whisper_kv_cache` struct we see that is has the following
fields:
```c++
struct whisper_kv_cache {
    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<whisper_kv_cell> cells;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    ggml_backend_buffer_t buffer = nullptr;

    std::vector<uint8_t> ctx_buf;
};
```
Now, one thing that I did not understand was the `ctx_buf` field. I understand
that the cache itself if stored in teh `ggml_backend_buffer` but what is the
`ctx_buf` for?
Well if we continue looking at the init function we find:
```c++
    cache.ctx_buf.resize(2*ggml_tensor_overhead());
```
This resizing the `ctx_buf` to twice the overhead of a tensor:
```console
(gdb) p ggml_tensor_overhead()
$17 = 368
(gdb) p 2*ggml_tensor_overhead()
$18 = 736
```
Then a ggml context will be created:
```c++
    struct ggml_init_params params = {
        /*.mem_size   =*/ cache.ctx_buf.size(),
        /*.mem_buffer =*/ cache.ctx_buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
```
Then we create tensors for the key and value caches:
```c++
    cache.k = ggml_new_tensor_1d(ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(ctx, wtype, n_elements);

    cache.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ...

    ggml_free(ctx);
```
Now, `ggml_backend_alloc_ctx_tensors` will take the tensors that have been
created in the context. And notice that the `ggml_context` is then freed.
So having the `ctx_buf` is a way to store the tensors in the context and then
be able to allocate them in a backend.
Now, one thing to keep in mind is that even though we are freeing the context
the data, like metadata for the tensors are still stored in the `ctx_buf`. One
migth think that this is what cache.k and cache.v are for but notice that these
are just pointers, and they actually point to the data stored in the `ctx_buf`:
```console
(gdb) p cache.k
$61 = (ggml_tensor *) 0x555555f451a0

(gdb) p cache.ctx_buf.data()
$63 = (unsigned char *) 0x555555f45180 " "

(gdb) p 0x555555f451a0 - 0x555555f45180
$65 = 32
```
So we can see that the tensor is stored 32 bytes after the start of the `ctx_buf`.
```
(gdb) p cache.ctx_buf.data() + 32
$71 = (unsigned char *) 0x555555f451a0 "\001

(gdb) p *(ggml_tensor*)(cache.ctx_buf.data() + 32)
$74 = {type = GGML_TYPE_F16, buffer = 0x555555717320, ne = {1572864, 1, 1, 1}, nb = {2, 3145728, 3145728, 3145728},
  op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  view_src = 0x0, view_offs = 0, data = 0x7fffee174040, name = '\000' <repeats 63 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}

(gdb) p *cache.k
$75 = {type = GGML_TYPE_F16, buffer = 0x555555717320, ne = {1572864, 1, 1, 1}, nb = {2, 3145728, 3145728, 3145728},
  op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  view_src = 0x0, view_offs = 0, data = 0x7fffee174040, name = '\000' <repeats 63 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```

Next we have the cross-attention cache which is for the output of the encoder
and is used by the decoder.
```c++
    if (!whisper_kv_cache_init(state->kv_cross, state->backends[0], ctx->itype,
                ctx->model.hparams.n_text_state,
                ctx->model.hparams.n_text_layer,
                GGML_PAD(ctx->model.hparams.n_audio_ctx, 256))) {
        WHISPER_LOG_ERROR("%s: whisper_kv_cache_init() failed for cross-attention cache\n", __func__);
        whisper_free_state(state);
        return nullptr;
    }
```
The difference here is that we are passing in a different cache, `kv_cross` and
also a different context size, `n_audio_ctx` instead of `n_text_ctx`.
```console
(gdb) p ctx->model.hparams.n_audio_ctx
$78 = 1500
```

Again we have 6 layers. But this time each layer need to store/cache up to n_ctx
(1536) tokens (1500 tokens padded to multiples of 256).
```
Key cache:
           <- n_text_state   ->
Layer 0: 0 [0              512]   ^                  Per layer: 512 * 1536 = 786432
                  ...             |
                  ...             n_ctx
                  ...             |
                  ...             |
                  ...             |
                  ...             |
      1535 [0              512]   v
...
...

Layer 5: 0 [0              512]   ^
                  ...             |
                  ...             n_ctx
                  ...             |
                  ...             |
                  ...             |
                  ...             |
      1535 [0              512]   v

Total: 6 * 786432 = 4718592 (values of some type like float16 etc)

And we have the same for the Value cache
Total: 6 * 786432 = 4718592

Key + Value = 9437184

And if we have 2 bytes per element/value:
9437184 * 2 = 18874368 bytes (18MB)
```

After that we have the `kv_pad` cache:
```c++
    if (!whisper_kv_cache_init(state->kv_pad, state->backends[0], ctx->itype,
                ctx->model.hparams.n_audio_state,
                1,
                GGML_PAD(ctx->model.hparams.n_audio_ctx, 256))) {
        WHISPER_LOG_ERROR("%s: whisper_kv_cache_init() failed for self-attention cache\n", __func__);
        whisper_free_state(state);
        return nullptr;
    }
```
Now this is interesting, we are passing in `n_audio_state` instead of
`n_text_state`, and also `1` instead of `n_text_layer`.
```console
(gdb) p ctx->model.hparams.n_audio_state
$86 = 512
```
So in this case there will only be a single layer:
```

Layer 0: 
        0 [0                1536]
                  ...
                  ...
                  ...
      511 [0                1536]
```
_wip_

### Timestamps
In whisper.cpp timestamps are mostly represented using `unit64_t` types which
provides exact integer precision without floating-point rounding errors. When
working on the VAD support I used floats/doubles initially to represent times
which partly because I though that would be exposed externally and using
floating point values would be more convenient.
Integer operations are generally faster than floating-point operations too
especially on systems that don't have floating point units. uint64_t is 8 bytes
so it is the same size as a double but it is more cache friendly. Also floating
point arithmetic can vary slightly between different CPU architectures and
compilers.
