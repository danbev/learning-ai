## Whisper Word Error Rate (WER) Tests
This page contains not related to implementing WER tests in whisper.cpp.

TODO: Add link to wer.md

### Alterative implementation
So we currently have whisper-cli which enables us to pass in multiple audio
files and have them transcribed. This program has a lot of options which is
great so that different configuration can be tested. Duplicating this would make
maintenance harder. But a program that delegates to whisper-cli would be nice
and it could in addition to the input audio files accept corresponding reference
text (the known correct transcription).

So with whisper-cli it is possible to provide multiple files and it is also
possible to use the option `--output-txt` to save the transcription to files
named after the input audio files. This really provides all that is needed for
the WER tests. The cli could accepts transcriptions and actual texts and the
calculate a score. This does not really involve whisper in this case but perhaps
having this in the whisper repository would be nice. Also having an interface
for this might be useful other examples in whisper.

### Enable Tests
Currently the tests are commented out in CMakeLists.txt.

First enable tests in CMakeLists.txt:
```console
$ cmake -S . -B build -DWHISPER_BUILD_TESTS=ON
$ cmake --build build
```
```console
$ ctest -R test-whisper-cli-tiny --test-dir build
```
To get verbose output, like console logging, use `-VV`.


### Unit tests
This is just an example of how unit tests might be created.

Then test can be run using `--test-dir` to avoid having to change directory:
```console
$ ctest --test-dir build -R test-assert
Internal ctest changing into directory: /home/danbev/work/ai/whisper-work/build
Test project /home/danbev/work/ai/whisper-work/build
    Start 10: test-assert
1/1 Test #10: test-assert ......................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
unit    =   0.00 sec*proc (1 test)

Total Test time (real) =   0.00 sec
```

Running via cmake:
```console

```

List available tests:
```console
$ ctest --test-dir build -N
Internal ctest changing into directory: /home/danbev/work/ai/whisper-work/build
Test project /home/danbev/work/ai/whisper-work/build
  Test  #1: test-whisper-cli-tiny
  Test  #2: test-whisper-cli-tiny.en
  Test  #3: test-whisper-cli-base
  Test  #4: test-whisper-cli-base.en
  Test  #5: test-whisper-cli-small
  Test  #6: test-whisper-cli-small.en
  Test  #7: test-whisper-cli-medium
  Test  #8: test-whisper-cli-medium.en
  Test  #9: test-whisper-cli-large
  Test #10: test-assert

Total Tests: 10
```


### Datasets
Open Speech and Language Resources ([OpenSLR](https://www.openslr.org/index.html))
is a project that provides


https://www.openslr.org/12

Lets take a look at one of these datasets:
https://www.openslr.org/resources/12/test-clean.tar.gz


If we look into the data set there is a README file which describes the data
set and is good to read.

If we look in the LibriSpeech/test-clean directory:
```console
$ ls test-clean/
1089  121   1284  1580  2094  237  2830  3570  3729  4446  4970  5105  5639  61   6829  7021  7176  8224  8455  8555
1188  1221  1320  1995  2300  260  2961  3575  4077  4507  4992  5142  5683  672  6930  7127  7729  8230  8463  908
```
These are directories for each speaker (thier id).
```console
$ cd test-clean/61/
$ ls
70968  70970
```
The above are identifiers of the chapters read by speaker 61.
```console
$ ls
61-70968-0000.flac  61-70968-0011.flac  61-70968-0022.flac
...
61-70968.trans.txt
```
