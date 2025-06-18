## Diarize issue
The version that worked for the reporter was v1.7.4 and the correct output is:
```
[00:00:00.000 --> 00:00:01.340]  (speaker 1) [sonnerie]
[00:00:01.340 --> 00:00:05.600]  (speaker 1) [silence]
[00:00:05.600 --> 00:00:08.880]  (speaker 1) — Salut, jeune homme. — C'est vrai que je te dérange ?
[00:00:08.880 --> 00:00:10.480]  (speaker 1) — Ah pas du tout, pas du tout, pas du tout.
[00:00:10.480 --> 00:00:14.020]  (speaker 1) J'étais en train de préparer un courrier.
```

Issue: https://github.com/ggml-org/whisper.cpp/issues/3092#issuecomment-2982050709

### Bisecting
```
$ git bisect start
$ git bisect bad
$ git bisect good v1.7.4
Bisecting: 357 revisions left to test after this (roughly 9 steps)
[77e0c86ab62eda9392a8567f4c29ab8d140cb0ba] whisper.wasm : fix unknown language issue (#3000)
```
So we build and test:
```console
$ ./build-debug.sh
$ ./run-diarize.sh
...
[00:00:00.000 --> 00:00:03.000]  (speaker 1) Sous-titres réalisés para la communauté d'Amara.org
```
This is not the expected output, so we continue bisecting:
```console
$ git bisect bad
Bisecting: 178 revisions left to test after this (roughly 8 steps)
[992b51b3d523e1442048d536272387521b4f5aa2] ggml: aarch64: implement SVE kernels for q2_k_q8_k vector dot (llama/12064)
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:02.700]  (speaker 1) *Bruit de réveil*
```
This is not the expected output, so we continue bisecting:
```console
$ git bisect bad
Bisecting: 88 revisions left to test after this (roughly 7 steps)
[3f91832352b2aca890102dc7ebc182f7fa095151] talk-llama : sync llama.cpp
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:01.340]  (speaker 1) [sonnerie]
[00:00:01.340 --> 00:00:05.600]  (speaker 1) [silence]
[00:00:05.600 --> 00:00:08.880]  (speaker 1) — Salut, jeune homme. — C'est vrai que je te dérange ?
[00:00:08.880 --> 00:00:10.480]  (speaker 1) — Ah pas du tout, pas du tout, pas du tout.
[00:00:10.480 --> 00:00:14.020]  (speaker 1) J'étais en train de préparer un courrier.
```
This matches the expected output, so we can stop bisecting:
```
[00:00:00.000 --> 00:00:01.340]  (speaker 1) [sonnerie]
[00:00:01.340 --> 00:00:05.600]  (speaker 1) [silence]
[00:00:05.600 --> 00:00:08.880]  (speaker 1) — Salut, jeune homme. — C'est vrai que je te dérange ?
[00:00:08.880 --> 00:00:10.480]  (speaker 1) — Ah pas du tout, pas du tout, pas du tout.
[00:00:10.480 --> 00:00:14.020]  (speaker 1) J'étais en train de préparer un courrier.
```
So we mark this as good:
```console
$ git bisect good
Bisecting: 44 revisions left to test after this (roughly 6 steps)
[e22d69839d530175327abc2158c5f56871d0d9c8] vulkan: linux builds + small subgroup size fixes (llama/11767)
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:01.340]  (speaker 1) [sonnerie]
[00:00:01.340 --> 00:00:05.600]  (speaker 1) [silence]
[00:00:05.600 --> 00:00:08.880]  (speaker 1) — Salut, jeune homme. — C'est vrai que je te dérange ?
[00:00:08.880 --> 00:00:10.480]  (speaker 1) — Ah pas du tout, pas du tout, pas du tout.
[00:00:10.480 --> 00:00:14.020]  (speaker 1) J'étais en train de préparer un courrier.
```
And this is also good:
```console
$ git bisect good
Bisecting: 22 revisions left to test after this (roughly 5 steps)
[4b60ff4f92bd4a767d5aff693484dc8255ec7672] metal : copy kernels for quant to F32/F16 conversions (llama/12017)
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:01.340]  (speaker 1) [sonnerie]
[00:00:01.340 --> 00:00:05.600]  (speaker 1) [silence]
[00:00:05.600 --> 00:00:08.880]  (speaker 1) — Salut, jeune homme. — C'est vrai que je te dérange ?
[00:00:08.880 --> 00:00:10.480]  (speaker 1) — Ah pas du tout, pas du tout, pas du tout.
[00:00:10.480 --> 00:00:14.020]  (speaker 1) J'étais en train de préparer un courrier.
```
```console
$ git bisect good
Bisecting: 11 revisions left to test after this (roughly 4 steps)
[c774eec709d153b94be60ebec8c7cb97f3bd82cd] go : improve model download (#2756)
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:02.700]  (speaker 1) *Bruit de réveil*
```
```console
$ git bisect bad
Bisecting: 5 revisions left to test after this (roughly 3 steps)
[9f83f67221814dab0477c3970fa5f618ac1e2a2b] common :  fix build min/max (#2845)
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:02.700]  (speaker 1) *Bruit de réveil*
```
```console
$ git bisect bad
Bisecting: 2 revisions left to test after this (roughly 1 step)
[17addf7104547a5d987a75fd35e1c86563c69f6c] sync : ggml
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:01.340]  (speaker 1) [sonnerie]
[00:00:01.340 --> 00:00:05.600]  (speaker 1) [silence]
[00:00:05.600 --> 00:00:08.880]  (speaker 1) — Salut, jeune homme. — C'est vrai que je te dérange ?
[00:00:08.880 --> 00:00:10.480]  (speaker 1) — Ah pas du tout, pas du tout, pas du tout.
[00:00:10.480 --> 00:00:14.020]  (speaker 1) J'étais en train de préparer un courrier.
```
```console
$ git bisect good
Bisecting: 0 revisions left to test after this (roughly 1 step)
[7d3da68f792018e81a758881e081154d1cbe6b6f] examples : use miniaudio for direct decoding flac, mp3, ogg and wav (#2759)
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:02.700]  (speaker 1) *Bruit de réveil*
```
```console
$ git bisect bad
Bisecting: 0 revisions left to test after this (roughly 0 steps)
[b5d21359c11fc9d19f8efb7bdcb0688d6b643d58] stream : stop on ^C when no audio is received (#2822)
```
```console
$ ./build-debug.sh
$ ./run-diarize.sh
[00:00:00.000 --> 00:00:01.340]  (speaker 1) [sonnerie]
[00:00:01.340 --> 00:00:05.600]  (speaker 1) [silence]
[00:00:05.600 --> 00:00:08.880]  (speaker 1) — Salut, jeune homme. — C'est vrai que je te dérange ?
[00:00:08.880 --> 00:00:10.480]  (speaker 1) — Ah pas du tout, pas du tout, pas du tout.
[00:00:10.480 --> 00:00:14.020]  (speaker 1) J'étais en train de préparer un courrier.
```
#### First bad commit
```console
$ git bisect good
7d3da68f792018e81a758881e081154d1cbe6b6f is the first bad commit
commit 7d3da68f792018e81a758881e081154d1cbe6b6f
Author: Dmitry Atamanov <data-man@users.noreply.github.com>
Date:   Thu Feb 27 12:06:54 2025 +0500

    examples : use miniaudio for direct decoding flac, mp3, ogg and wav (#2759)

 Makefile                      |    13 +-
 examples/addon.node/addon.cpp |     4 +-
 examples/cli/cli.cpp          |     9 +-
 examples/common.cpp           |   164 +-
 examples/common.h             |     2 +-
 examples/dr_wav.h             |  8815 -----
 examples/generate-karaoke.sh  |     9 +-
 examples/miniaudio.h          | 93468 ++++++++++++++++++++++++++++++++++++++++++++++++++++
 examples/server/server.cpp    |    10 +-
 examples/stb_vorbis.c         |  5584 ++++
 10 files changed, 99149 insertions(+), 8929 deletions(-)
 delete mode 100644 examples/dr_wav.h
 create mode 100644 examples/miniaudio.h
 create mode 100644 examples/stb_vorbis.c
 ```

#### Bisect log
 ```console
 $ git bisect log
git bisect start
# status: waiting for both good and bad commits
# bad: [f3ff80ea8da044e5b8833e7ba54ee174504c518d] examples : set the C++ standard to C++17 for server (#3261)
git bisect bad f3ff80ea8da044e5b8833e7ba54ee174504c518d
# status: waiting for good commit(s), bad commit known
# good: [8a9ad7844d6e2a10cddf4b92de4089d7ac2b14a9] release : v1.7.4
git bisect good 8a9ad7844d6e2a10cddf4b92de4089d7ac2b14a9
# bad: [77e0c86ab62eda9392a8567f4c29ab8d140cb0ba] whisper.wasm : fix unknown language issue (#3000)
git bisect bad 77e0c86ab62eda9392a8567f4c29ab8d140cb0ba
# bad: [992b51b3d523e1442048d536272387521b4f5aa2] ggml: aarch64: implement SVE kernels for q2_k_q8_k vector dot (llama/12064)
git bisect bad 992b51b3d523e1442048d536272387521b4f5aa2
# good: [3f91832352b2aca890102dc7ebc182f7fa095151] talk-llama : sync llama.cpp
git bisect good 3f91832352b2aca890102dc7ebc182f7fa095151
# good: [e22d69839d530175327abc2158c5f56871d0d9c8] vulkan: linux builds + small subgroup size fixes (llama/11767)
git bisect good e22d69839d530175327abc2158c5f56871d0d9c8
# good: [4b60ff4f92bd4a767d5aff693484dc8255ec7672] metal : copy kernels for quant to F32/F16 conversions (llama/12017)
git bisect good 4b60ff4f92bd4a767d5aff693484dc8255ec7672
# bad: [c774eec709d153b94be60ebec8c7cb97f3bd82cd] go : improve model download (#2756)
git bisect bad c774eec709d153b94be60ebec8c7cb97f3bd82cd
# bad: [9f83f67221814dab0477c3970fa5f618ac1e2a2b] common :  fix build min/max (#2845)
git bisect bad 9f83f67221814dab0477c3970fa5f618ac1e2a2b
# good: [17addf7104547a5d987a75fd35e1c86563c69f6c] sync : ggml
git bisect good 17addf7104547a5d987a75fd35e1c86563c69f6c
# bad: [7d3da68f792018e81a758881e081154d1cbe6b6f] examples : use miniaudio for direct decoding flac, mp3, ogg and wav (#2759)
git bisect bad 7d3da68f792018e81a758881e081154d1cbe6b6f
# good: [b5d21359c11fc9d19f8efb7bdcb0688d6b643d58] stream : stop on ^C when no audio is received (#2822)
git bisect good b5d21359c11fc9d19f8efb7bdcb0688d6b643d58
# first bad commit: [7d3da68f792018e81a758881e081154d1cbe6b6f] examples : use miniaudio for direct decoding flac, mp3, ogg and wav (#2759)
```

### Troubleshooting
So this change was introduced in [#2759](https://github.com/ggml-org/whisper.cpp/pull/2759).

```console
$ git bisect reset
```

So looking at the first bad commit:
```console
$ git show 7d3da68f792018e81a758881e081154d1cbe6b6f --stat
commit 7d3da68f792018e81a758881e081154d1cbe6b6f
Author: Dmitry Atamanov <data-man@users.noreply.github.com>
Date:   Thu Feb 27 12:06:54 2025 +0500

    examples : use miniaudio for direct decoding flac, mp3, ogg and wav (#2759)

 Makefile                      |    13 +-
 examples/addon.node/addon.cpp |     4 +-
 examples/cli/cli.cpp          |     9 +-
 examples/common.cpp           |   164 +-
 examples/common.h             |     2 +-
 examples/dr_wav.h             |  8815 -----
 examples/generate-karaoke.sh  |     9 +-
 examples/miniaudio.h          | 93468 ++++++++++++++++++++++++++++++++++++++++++++++++++++
 examples/server/server.cpp    |    10 +-
 examples/stb_vorbis.c         |  5584 ++++
 10 files changed, 99149 insertions(+), 8929 deletions(-)
```
Since the commit was merged there has been some refactoring of the audio decoding and most
if not all of it has been moved to `examples/whisper-common.cpp`.

One thing to keep in mind is that this might not just be related to diarization but also
stereo audio decoding which there have been issues opened related to that as well.

```console
$ git show 7d3da68f -- examples/common.cpp
```

MiniAudio will upmix mono to stereo (duplicating samples), or downmix stereo to mono
(averaging), without telling you which means we can’t know if you’re processing actual
stereo or just duplicated mono.

Looking at the timestamps, first from master:
```console
[00:00:00.000 --> 00:00:03.000]  (speaker 1) Sous-titres réalisés para la communauté d'Amara.org
```
And then the working:
```console
[00:00:00.000 --> 00:00:01.340]  (speaker 1) [sonnerie]
[00:00:01.340 --> 00:00:05.600]  (speaker 1) [silence]
[00:00:05.600 --> 00:00:08.880]  (speaker 1) — Salut, jeune homme. — C'est vrai que je te dérange ?
[00:00:08.880 --> 00:00:10.480]  (speaker 1) — Ah pas du tout, pas du tout, pas du tout.
[00:00:10.480 --> 00:00:14.020]  (speaker 1) J'étais en train de préparer un courrier.
```

I've verified that the sample audio has two channels both by inspecting and listening
to it in Audacity, but also printing like this:
```console
read_audio_data: audio data has 2 channel(s)
```

So in `whisper-cli` we have the following:
```c++
        if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {
            fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp.c_str());
            continue;
        }
```
This will call into `whisper-common.cpp`:
```c++
bool read_audio_data(const std::string & fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo) {
    ...
    // Added by danbev
    uint32_t actual_channels = decoder.outputChannels;
    fprintf(stderr, "%s: audio data has %u channel(s)\n", __func__, actual_channels);
    if (stereo && actual_channels != 2) {
        fprintf(stderr, "Error: requested stereo, but input is %u channel(s)\n", actual_channels);
        ma_decoder_uninit(&decoder);
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;

    if ((result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count)) != MA_SUCCESS) {
		fprintf(stderr, "error: failed to retrieve the length of the audio data (%s)\n", ma_result_description(result));

		return false;
    }

    pcmf32.resize(stereo ? frame_count*2 : frame_count);
}
```
```console
(lldb) br set -f common-whisper.cpp -l 114
(lldb) p pcmf32.size()
(std::vector<float>::size_type) 0
(lldb) p stereo
(bool) true
(lldb) p frame_count
(ma_uint64) 224256
(lldb) n
(lldb) p pcmf32.size()
(std::vector<float>::size_type) 448512
```
Next we are going to read frames using this call:
```c++
    if ((result = ma_decoder_read_pcm_frames(&decoder, pcmf32.data(), frame_count, &frames_read)) != MA_SUCCESS) {
		fprintf(stderr, "error: failed to read the frames of the audio data (%s)\n", ma_result_description(result));

		return false;
    }
```
And notice the size of `pcmf32` and the `frame_count`:
```console
(lldb) p pcmf32.size()
(std::vector<float>::size_type) 448512
(lldb) p frame_count
(ma_uint64) 224256
```
After the call the number of frames read is:
```console
(lldb) p frames_read
(ma_uint64) 224256
```
And the final part of the function looks like this:
```c++
    if (stereo) {
        fprintf(stderr, "%s: processing stereo audio data.......\n", __func__);
		pcmf32s.resize(2);
		pcmf32s[0].resize(frame_count);
		pcmf32s[1].resize(frame_count);
		for (uint64_t i = 0; i < frame_count; i++) {
			pcmf32s[0][i] = pcmf32[2*i];
			pcmf32s[1][i] = pcmf32[2*i + 1];
		}
    }

    ma_decoder_uninit(&decoder);

    return true;
}
```
So that is resizing `pcmf32s` to 2 channels and then copying the data from `pcmf32` into
the two channels.
But the number of frames read is just 224256 and each of the vectors of `pcmf32s` is
224256, so we have 224256 * 2 = 448512 which is the size of `pcmf32` but only half
of that number of frames has been read.

If we look at the working version in `commmon.cpp`:
```c++
718     std::vector<int16_t> pcm16;
719     pcm16.resize(n*wav.channels);
720     drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
721     drwav_uninit(&wav);
722
723     // convert to mono, float
724     fprintf(stderr, "[danbev], n == %llu, wav.channels == %d\n", n, wav.channels);
725     fprintf(stderr, "[danbev], pcm16.size() == %zu\n", pcm16.size());
726     pcmf32.resize(n);
727     if (wav.channels == 1) {
728         for (uint64_t i = 0; i < n; i++) {
729             pcmf32[i] = float(pcm16[i])/32768.0f;
730         }
731     } else {
732         for (uint64_t i = 0; i < n; i++) {
733             pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
734         }
735     }
736
737     if (stereo) {
738         // convert to stereo, float
739         pcmf32s.resize(2);
740
741         pcmf32s[0].resize(n);
742         pcmf32s[1].resize(n);
743         for (uint64_t i = 0; i < n; i++) {
744             pcmf32s[0][i] = float(pcm16[2*i])/32768.0f;
745             pcmf32s[1][i] = float(pcm16[2*i + 1])/32768.0f;
746         }
747     }
```
Notice that `pcmf32` is updated with the interleaved channels, and this is missing
from the current implementation.
So the previous version would read stereo int16 data -> pcm16 (448512 samples) and
then convert to mono (224256 samples), and also convert to stereo in `pcmf32s`

The current code is missing the conversion to mono, so it reads stereo float data
(448512 samples) and then converts to stereo in `pcmf32s` (224256 samples each).


_wip_


```console
$ git checkout 7d3da68f^
HEAD is now at b5d21359 stream : stop on ^C when no audio is received (#2822)
```
That should pass which it does, so we can then checkout the failing commit:
```console
$ git checkout 7d3da68f
Previous HEAD position was b5d21359 stream : stop on ^C when no audio is received (#2822)
HEAD is now at 7d3da68f examples : use miniaudio for direct decoding flac, mp3, ogg and wav (#2759)
```
