### VAD Segments Repeat Issue
This document describes an issue with the VAD segments repeating in the output
of the `whisper_cli` in combination with using VAD.

Issue: https://github.com/ggml-org/whisper.cpp/issues/3162

### Investigation
Running this locally and also following along in audacity I'm seeing the following
output:
```console
./build/bin/whisper-cli -vm models/ggml-silero-v5.1.2.bin --vad -f samples/transcribe.wav -m models/ggml-large-v3.bin -osrt
...
00:37:53,790 --> 00:38:00,540
 which is here because let me first uh free a little bit more of this portion of the kidney

83
00:38:00,540 --> 00:38:16,570
 so yeah absolutely yeah yeah so that is that is in my opinion what i would do the ultrasound

84
00:38:16,570 --> 00:38:22,310
 um i don't know have you ever have you ever used the icg guys for

85
00:38:22,310 --> 00:38:25,290
 to identify the thrombus this is interesting i've never used it

86
00:38:34,230 --> 00:38:36,160
 um i don't know if i've ever used the icg guys for

87
00:38:36,160 --> 00:38:42,060
 um i don't know if i've ever used the icg guys for

88
00:38:42,060 --> 00:38:44,300
 um i don't know if i've ever used the icg guys for

89
00:38:44,300 --> 00:39:04,440
 um i don't know if i've ever used the icg guys for
```
Notice how these segments
```console
84
00:38:16,570 --> 00:38:22,310
 um i don't know have you ever have you ever used the icg guys for

85
00:38:22,310 --> 00:38:25,290
 to identify the thrombus this is interesting i've never used it

86
00:38:34,230 --> 00:38:36,160
 um i don't know if i've ever used the icg guys for
```
Notice that there is a gap between 85 and 86 in the time stamps.

So let start by saving the samples that are passed into whipser after the
vad processing to actaully listen to what the file sounds like. I've saved
the `vad_samples` to a file named `vad_samples.wav` and listened to it: 
```console
$ play vad_samples.wav trim 38:00

vad_samples.wav:

 File Size: 114M      Bit Rate: 256k
  Encoding: Signed PCM
  Channels: 1 @ 16-bit
Samplerate: 16000Hz
Replaygain: off
  Duration: 00:59:27.39

In:64.1% 00:38:08.19 [00:21:19.20] Out:356k  [     =|=     ]        Clip:0
```
I've listened to different parts of the `vad_samples.wav` file and I can't hear
any repeating pattern, in fact the output sounds clear.

```console
$ ls -lh vad_samples.wav
-rw-r--r--@ 1 danbev  staff   109M May 19 09:26 vad_samples.wav
$ ls -lh samples/transcribe.wav
-rw-r--r--@ 1 danbev  staff   277M May 19 08:04 samples/transcribe.wav
````

So it looks like that vad processing step is working as expected. But there
is also a portion of the VAD processing that is performed when `whisper_cli`
calls `whisper_full_get_segment_t0` and `whisper_full_get_segment_t1` to get
the timestamp for a transcribed segment:
```c++
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }
```
But this is done after whisper has transcribed the input which is our
case is the output of the VAD processing.

Lets try without VAD enabled by removing the `--vad` flag and see if we get
the same/similar output. It wont be the same as this time all audio will be
processed, even non-speech.
```console
[00:00:00.800 --> 00:00:02.260]   Good morning everyone.
[00:00:02.260 --> 00:00:07.760]   Today I'm going to present a case of robot assisted right nephrourethrectomy performed
[00:00:07.760 --> 00:00:10.360]   by Dr. Alberto Breda.
[00:00:10.360 --> 00:00:16.720]   The patient is a 58 years old male, active smoker with previous tonsillectomy.
[00:00:16.720 --> 00:00:22.440]   He referred gross hematuria and weight loss in the last weeks, so he performed a total
[00:00:22.440 --> 00:00:29.000]   body CT scan that showed a large mass of the right kidney associated to hydronephrosis
[00:00:29.000 --> 00:00:33.100]   and to a mass inside the distal part of the ureter.
[00:00:33.100 --> 00:00:40.000]   The lymph nodes were negative, but the patient had multiple abnormal veins coming out from
[00:00:40.000 --> 00:00:41.720]   the inferior vena cava.
[00:00:41.720 --> 00:00:47.360]   The chest was clear as well as the preoperative cystoscopy.
[00:00:47.360 --> 00:00:58.000]   Here we have the imaging of the CT scan.
[00:00:58.000 --> 00:01:13.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:13.000 --> 00:01:15.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:15.000 --> 00:01:15.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:15.000 --> 00:01:16.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:16.000 --> 00:01:16.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:16.000 --> 00:01:16.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:16.000 --> 00:01:16.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:16.000 --> 00:01:16.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:16.000 --> 00:01:16.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:16.000 --> 00:01:16.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:16.000 --> 00:01:18.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:18.000 --> 00:01:18.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:18.000 --> 00:01:20.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:20.000 --> 00:01:20.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:20.000 --> 00:01:34.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:34.000 --> 00:01:34.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:34.000 --> 00:01:34.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:34.000 --> 00:01:34.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:34.000 --> 00:01:34.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:34.000 --> 00:01:34.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:34.000 --> 00:01:44.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:44.000 --> 00:01:44.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:01:44.000 --> 00:02:02.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:02:02.000 --> 00:02:02.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:02:02.000 --> 00:02:02.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:02:02.000 --> 00:02:02.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:02:02.000 --> 00:02:02.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:02:02.000 --> 00:02:02.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:02:02.000 --> 00:02:02.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:02:02.000 --> 00:02:02.000]   And here we have the 3D reconstruction of the right kidney thanks to MEDx program.
[00:02:02.000 --> 00:02:10.720]   so close. He must have had an appendicitomy with peritonitis.
[00:02:32.000 --> 00:02:39.160]   Yes, it's not fat, but it's fat. There's a nice moment represented.
[00:02:39.160 --> 00:02:47.880]   I'll hold the camera, otherwise you won't be able to.
[00:02:47.880 --> 00:03:01.800]   But because we're doing, we really have
[00:03:01.800 --> 00:03:01.920]   a lot of work to do.
[00:03:01.920 --> 00:03:01.980]   We're doing a lot of work to do.
[00:03:02.000 --> 00:03:05.180]   Next week there will be the National Congress.
[00:03:05.180 --> 00:03:12.560]   This week we were all in Amsterdam for the AU, the meetings.
[00:03:12.560 --> 00:03:21.200]   There's a lot of people, but you know, they also have to work at home, otherwise we won't be able to go on with the lists.
[00:03:32.000 --> 00:03:35.180]   So we're going to be doing a lot of work, but you know, we're going to be doing a lot of work, but you know, they also have to work at home, otherwise we won't be able to go on with the lists.
[00:03:35.180 --> 00:03:35.240]   We're going to be doing a lot of work, but you know, they also have to work at home, otherwise we won't be able to go on with the lists.
[00:04:02.000 --> 00:04:31.980]   Thank you.
[00:04:32.000 --> 00:05:01.980]   Thank you.
[00:05:02.000 --> 00:05:31.980]   Thank you.
[00:05:32.000 --> 00:06:01.980]   Thank you.
[00:06:02.000 --> 00:06:31.980]   Thank you.
[00:06:32.000 --> 00:07:01.980]   Thank you.
[00:07:02.000 --> 00:07:31.980]   Thank you.
[00:07:32.000 --> 00:08:01.980]   Thank you.
[00:08:02.000 --> 00:08:31.980]   Thank you.
[00:08:32.000 --> 00:09:01.980]   Thank you.
[00:09:02.000 --> 00:09:31.980]   Thank you.
[00:09:32.000 --> 00:10:01.980]   Thank you.
[00:10:02.000 --> 00:10:31.980]   Thank you.
[00:10:32.000 --> 00:11:01.980]   Thank you.
[00:11:02.000 --> 00:11:03.060]   Thank you.
[00:11:03.060 --> 00:11:04.000]   Thank you.
[00:11:04.000 --> 00:11:05.060]   Thank you.
[00:11:05.060 --> 00:11:06.100]   Thank you.
[00:11:06.100 --> 00:11:07.100]   Thank you.
[00:11:07.100 --> 00:11:08.100]   Thank you.
[00:11:08.100 --> 00:11:09.100]   Thank you.
[00:11:09.100 --> 00:11:10.100]   Thank you.
[00:11:10.100 --> 00:11:11.100]   Thank you.
[00:11:11.100 --> 00:11:12.100]   Thank you.
[00:11:12.100 --> 00:11:13.100]   Thank you.
[00:11:13.100 --> 00:11:14.100]   Thank you.
[00:11:14.100 --> 00:11:15.100]   Thank you.
[00:11:15.100 --> 00:11:16.100]   Thank you.
[00:11:16.100 --> 00:11:17.100]   Thank you.
[00:11:17.100 --> 00:11:18.100]   Thank you.
[00:11:18.100 --> 00:11:19.100]   Thank you.
[00:11:19.100 --> 00:11:20.100]   Thank you.
[00:11:20.100 --> 00:11:21.100]   Thank you.
[00:11:21.100 --> 00:11:22.100]   Thank you.
[00:11:22.100 --> 00:11:23.100]   Thank you.
[00:11:23.100 --> 00:11:24.100]   Thank you.
[00:11:24.100 --> 00:11:25.100]   Thank you.
[00:11:25.100 --> 00:11:26.100]   Thank you.
[00:11:26.100 --> 00:11:27.100]   Thank you.
[00:11:27.100 --> 00:11:28.100]   Thank you.
[00:11:28.100 --> 00:11:29.140]   Thank you.
[00:11:29.140 --> 00:11:30.180]   Thank you.
[00:11:30.180 --> 00:11:31.220]   Thank you.
[00:11:31.220 --> 00:11:31.860]   Thank you.
[00:11:31.880 --> 00:11:32.880]   Thank you.
[00:11:32.880 --> 00:11:33.880]   Thank you.
```
So we get a lot of repeats without VAD as well.
