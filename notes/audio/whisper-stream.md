###
First we create an instance of the audio_async struct which is defined in
common-sdl.h:
```c++
    audio_async audio(params.length_ms);
```
The `length_ms` parameter determines the total duration of the audio buffer in
milliseconds. This controls how many seconds of recent audio are kept in memory
(e.g., 10000 ms = 10 seconds of audio history). 



```c++
class audio_async {
public:
    audio_async(int len_ms);
    ~audio_async();

    bool init(int capture_id, int sample_rate);
    ...

private:
    SDL_AudioDeviceID m_dev_id_in = 0;

    int m_len_ms = 0;
    int m_sample_rate = 0;

    std::atomic_bool m_running;
    std::mutex       m_mutex;

    std::vector<float> m_audio;
    size_t             m_audio_pos = 0;
    size_t             m_audio_len = 0;
};
```
The `m_audio` is the circular buffer and `m_audio_pos` is the current position
in the buffer. The `m_audio_len` is number of valid samples in the buffer.

```c++
audio_async::audio_async(int len_ms) {
    m_len_ms = len_ms;

    m_running = false;
}
```
And by default the length_ms is set to 10000ms (10 seconds):
```c++
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    ...
```

```c++
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }
```

```c++
bool audio_async::init(int capture_id, int sample_rate) {
    SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n", SDL_GetError());
        return false;
    }

    SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);
    ...
```
This is a hint to SDL to use "medium" for resampling. Most audio hardware sample
rates are 44.1kHz or 48kHz, so we can use a sample rate of 48kHz for the audio
but in whisper.cpp the sample rate is 16kHz (the model is trained on 16Hz, human
speach is up to 8kHz and this matches the nyquist frequency of 16kHz).
```c++
    SDL_AudioSpec capture_spec_requested;
    SDL_AudioSpec capture_spec_obtained;

    SDL_zero(capture_spec_requested);
    SDL_zero(capture_spec_obtained);

    capture_spec_requested.freq     = sample_rate;
    capture_spec_requested.format   = AUDIO_F32;
    capture_spec_requested.channels = 1;
    capture_spec_requested.samples  = 1024;
    capture_spec_requested.callback = [](void * userdata, uint8_t * stream, int len) {
        audio_async * audio = (audio_async *) userdata;
        audio->callback(stream, len);
    };
    capture_spec_requested.userdata = this;
```
Notice that the number off samlpes is set to 1024 and that is is in `samples`
and the actual size of the buffers that are passed to the callback will be
1024*AUDIO_F32 (4 bytes) = 4096 bytes.

Then the device is opened and the last thing to happen in the function is:
```c++
    m_sample_rate = capture_spec_obtained.freq;

    m_audio.resize((m_sample_rate*m_len_ms)/1000);

    return true;
}
```
This is calculating the size of the audio buffer based on the sample rate
and the desired length in milliseconds. The formula is:
```console
sampling rate * length in ms / 1000 = number of samples


(16000 * 10000  ) / 1000 = 160 000 samples
         (10 sec)
(16000 * 30000  ) / 1000 = 480 000 samples
        (30 sec)
```
So in our case the circular buffer will be 160000 samples long (640000 bytes, 10
seconds of audio samples).

When SDL captures new audio data it will call the callback function, and the
size of the `stream` buffer is passes is the sample size
(`capture_spec_requested.samples`) in bytes (4096 bytes).


```c++
// callback to be called by SDL
void audio_async::callback(uint8_t * stream, int len) {
    if (!m_running) {
        return;
    }

    size_t n_samples = len / sizeof(float);

    if (n_samples > m_audio.size()) {
        n_samples = m_audio.size();

        stream += (len - (n_samples * sizeof(float)));
    }
```
The above calculates the number of incoming samples in the stream buffer passed
from SDL is greater than the current size of the circular buffer, it limits the
number of incoming samples to the current size of the circular buffer.
```
len = 4096 bytes)
n_samples = 4096 / 4 = 1024 samples

Lets say the circular buffer is 1000 samples long
m_audio.size() = 1000

stream 
   ↓
  [0                                                    4095]

n_samples = 1000

(len - (n_samples * sizeof(float)))
4096 - (1000      * 4)
4096 - 96

             stream 
               ↓
  [0        96                                          4095]


Initial:
  stream
    ↓
  [Sample 0][Sample 1]...[Sample 23][Sample 24]...[Sample 1023]
   0        4            92          96           ...          4092-4095

After adjustment:
                                      stream
                                        ↓
  [Sample 0][Sample 1]...[Sample 23][Sample 24]...[Sample 1023]
   0        4            92          96           ...          4092-4095
```
So this is skipping the first 96 bytes of the stream buffer. And notice that this
is sample 24. By skipping the first 24 samples we are keeping the last 1000
samples.

Next we have will write the incoming samples to the circular buffer:
```c++
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_audio_pos + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - m_audio_pos;

            memcpy(&m_audio[m_audio_pos], stream, n0 * sizeof(float));
            memcpy(&m_audio[0], stream + n0 * sizeof(float), (n_samples - n0) * sizeof(float));
        } else {
            memcpy(&m_audio[m_audio_pos], stream, n_samples * sizeof(float));
        }

        m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
        m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
    }
}
```
First we lock the mutex to prevent other threads from accessing the audio buffer
. Is this alright to do in a callback from SDL when we don't know how long it
might take to get the lock?

The code is checking if the current write position of the circular buffer, 
`m_audio_pos`, plus the number of incoming samples, `n_samples`, is greater than
the size of the circular buffer. First `n0` is calculated which is the number of
samples that can be written to the buffer before it wraps around. Then we write
to the circular buffer by copying that number of samples from the incoming
stream buffer. After that we copy to the beginning of the circlar buffer, this
time starting from the incoming stream plus n0.

So after the callback completes `m_audio_pos` will be index of the next sample
to be written to the circular buffer. The `m_audio_len` is the number of valid
samples in the circular buffer.

### get (audio)
This function is used to get data from the circular buffer.
```c++
void audio_async::get(int ms, std::vector<float> & result) {
    ...

    result.clear();

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (ms <= 0) {
            ms = m_len_ms;
        }

        size_t n_samples = (m_sample_rate * ms) / 1000;
        if (n_samples > m_audio_len) {
            n_samples = m_audio_len;
        }

        result.resize(n_samples);
```

Now, recall that `m_audio_pos` is the index of the next sample to be written to
the so if we want to get the last `ms` milliseconds of audio we actually need
to access that number of samples backwards from the current position:
```c++

        int s0 = m_audio_pos - n_samples;
        if (s0 < 0) {
            s0 += m_audio.size();
        }
```
And if that values is 0 or negative we need to wrap around the circular buffer
and start from the beginning of the buffer. So this is like if close to the 
limit of the buffer then this above will place us at the correct index position
to read the start of `ms` samples.

And then we have the reading from the cicular buffer to the result buffer/vector:
```c++

        if (s0 + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - s0;

            memcpy(result.data(), &m_audio[s0], n0 * sizeof(float));
            memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(float));
        } else {
            memcpy(result.data(), &m_audio[s0], n_samples * sizeof(float));
        }
    }
}
```
And this is almost the same as the writing to the circular buffer.


Next, in `stream.cpp` we create three buffers:
```c++
    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
```
`pcmf32_new` recieves the new audio samples which is what we pass to
`audio_async::get()`.  `pcmf32_old` contains previous samples, and `pcmf32` is
the buffer that is passed to `whisper_full`.
```c++
    while (is_running) {
        ...
        if (!use_vad) {
            while (true) {
                // handle Ctrl + C
                is_running = sdl_poll_events();
                if (!is_running) {
                    break;
                }


    }
```
`sdl_poll_events()` looks like this:
```c++
bool sdl_poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                {
                    return false;
                }
            default:
                break;
        }
    }

    return true;
}
```
So this is polling SDL for events and only handling the quit event.
And after that we will copy audio data from the cirular buffer into `pcmf32_new`:
```c++
                audio.get(params.step_ms, pcmf32_new);
```
```c++
                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }
```
```c++
    const int n_samples_step = (1e-3*params.step_ms)*WHISPER_SAMPLE_RATE;
```
The default value for `step_ms` is 3000ms (3 seconds) and WHISPER_SAMPLE_RATE is
16000Hz. And `1e-3` is 0.001 so this is converting milliseconds to seconds. And
then finally multiplying by the sample rate to get the number of samples.
One way to think about this is that the user facing parameters are in time which
is more convenient for the user, but internally descrete samples are used in
processing (arrays or floating points numbers representing the amplitude).

Following that, we check if the size of the samples read from the cicular
buffer is greater than or equal to the the number of samples in a step, and if
so we call audio_sync::clear:
```c++
                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }
```
This is marking the cicular buffer as processed and ready for the new data.

```c++
bool audio_async::clear() {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to clear!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_audio_pos = 0;
        m_audio_len = 0;
    }

    return true;
}
```
Next we have:
```c++
            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));
```
So `n_samples_new` is the number of samples in the new audio buffer that was
read from the circular buffer. Lets look at these local variables and sort out
what they mean:
```c++
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    ...
    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;
```
So by default `keep_ms` will be 200ms, and `length_ms` will be 10000ms (10 sec)
and these will also be converted to number of samples like we saw before.
So `n_samples_len` will be 160000 samples (10 seconds) and `n_samples_keep` will
be 3200 samples (200ms). So the `n_samples_take` will be the minimum of the
```console
(gdb) p params
$5 = {n_threads = 4,
step_ms = 3000,
length_ms = 10000,
keep_ms = 200,

capture_id = -1,
max_tokens = 32, 
audio_ctx = 0,
beam_size = -1,
vad_thold = 0.600000024,
freq_thold = 100,
translate = false,
no_fallback = false, 
print_special = false,
no_context = true,
no_timestamps = false,
tinydiarize = false,
save_audio = false, 
use_gpu = true,
flash_attn = false,
language = "en",
model = "./models/ggml-base.en.bin", fname_out = ""}


(gdb) p n_samples_step 
$1 = 48000
(gdb) p n_samples_len 
$2 = 160000
(gdb) p n_samples_keep 
$3 = 3200
(gdb) p n_samples_30s 
$4 = 480000
```

Next we calculate the number of samples that we should carry forward from the
previous audio buffer `pcmf32_old`. We use an overlap to maintain context and
ensure words span buffer boundries.
```c++
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));
```
So we start with the full processning window sample size which is `n_samples_len`.
We then add the number of samples that we want to overlap (keep). And then
substract the number of new samples.
And we use this number to resize the `pcmf32` buffer which is the buffer that
will be passed to whisper:
```c++
            pcmf32.resize(n_samples_new + n_samples_take);
```
And then we copy from the old buffer `n_samples_take` samples to the beggining
of the `pcmf32` buffer:
```c++
            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }
```
And the we copy the new data to the location past the previous data copied above:
```c++
            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));
            pcmf32_old = pcmf32;
```
And we also set the old data to be point to the new data that is about to be
processed.
```c++
            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }
```
```c++
            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());
```
Notice that this is creating a new vector of the last `n_samples_keep` samples.

And we also need to update the prompt tokens for the next iteration:
```c++

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
            fflush(stdout);
```
