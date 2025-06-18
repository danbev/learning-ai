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
