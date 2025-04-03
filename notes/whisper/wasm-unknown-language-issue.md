## Whisper WASM App Unknown Language Issue

When accessing the app https://ggerganov.github.io/whisper.cpp/ and selecting
an non-English language model and then transcribing an audio file (which can
be in English) the following is displayed in the dev console:
```console
whisper_lang_id: unknown language 'д=␙c'
```
Now, the log message is coming from
```c++
int whisper_lang_id(const char * lang) {
    if (!g_lang.count(lang)) {
        for (const auto & kv : g_lang) {
            if (kv.second.second == lang) {
                return kv.second.first;
            }
        }

        WHISPER_LOG_ERROR("%s: unknown language '%s'\n", __func__, lang);
        return -1;
    }
    return g_lang.at(lang).first;
}
```

I'm not able to reproduce this issue locally using the Python server.

Lets add some logging of `lang` and run this locally:
```console
mkdir -p build-em
pushd build-em
emcmake cmake ..
make -j8
popd
python3 examples/server.py
```

After clearning all application data for this site, I loaded `tiny-q5_1` and
then selected `jfk.wav` and clicked `Transcribe`:
```console
js: full_default returned: 0
Endianness check: [04 03 02 01]
helpers.js:14 whisper_full_with_state: language pointer: 0x5a41b8, bytes: [65 6e 00 00]
helpers.js:14 whisper_full_with_state: [danbev] params.language: en
helpers.js:14 whisper_full_with_state: [danbev] after copy params.language: en
helpers.js:14 whisper_full_with_state: [danbev] not autodetecting....
helpers.js:14 whisper_full_with_state: [danbev] is multilingual params.language: en
helpers.js:14 whisper_full_with_state: language pointer: 0x5a41b8, bytes: [65 6e 00 00]
helpers.js:14 whisper_lang_id: [danbev] lang 'en'
helpers.js:14 [00:00:00.000 --> 00:00:11.000]   And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```
So locally this looks good and my environment is little endian.

Now, lets deploy this version with the logging to my fork and see if what the
output is.

The other CI jobs can be disabled which can be useful so that the complete CI
builds are not run:
```console
$ cat disable-jobs.sh 
git mv .github/workflows/bindings-go.yml .github/workflows/bindings-go.yml.disabled
git mv .github/workflows/bindings-ruby.yml .github/workflows/bindings-ruby.yml.disabled
git mv .github/workflows/build.yml .github/workflows/build.yml.disabled
git mv .github/workflows/examples.yml .github/workflows/examples.yml.disabled
git mv .github/workflows/docker.yml .github/workflows/docker.yml.disabled
```
Then I push to my fork's master branch:
```console
$ git push -f origin wasm-unknown-language-issue:master
```

The above will run the job to deploy the github pages to my fork:
https://danbev.github.io/whisper.cpp/

After clearning all application data for this site, I loaded `tiny-q5_1` and
then selected `jfk.wav` and clicked `Transcribe`:
```console
js: full_default returned: 0
helpers.js:14 Endianness check: [04 03 02 01]
helpers.js:14 whisper_full_with_state: language pointer: 0x5a4a8c, bytes: [60 71 3d 19]
helpers.js:14 whisper_full_with_state: [danbev] params.language: `q=␙c
helpers.js:14 whisper_full_with_state: [danbev] after copy params.language: `q=␙c
helpers.js:14 whisper_full_with_state: [danbev] not autodetecting....
helpers.js:14 whisper_full_with_state: [danbev] is multilingual params.language: `q=␙c
helpers.js:14 whisper_full_with_state: language pointer: 0x5a4a8c, bytes: [60 71 3d 19]
helpers.js:14 whisper_lang_id: [danbev] lang '`q=␙c'
helpers.js:14 whisper_lang_id: unknown language '0q=␙c'
helpers.js:14 [00:00:00.000 --> 00:00:11.000]   So my fellow Americans, ask not what your country can do for you as what you can do for your country.
```

Looking at the memory contents on the github pages version is does not look
correct at all. How could row before it print out `en` correctly from
those bytes? 
- Could this be a memory corruption issue? 
- an endian (little vs big) issue?

I've checked for endianness in the code and it seems to be correct and the output
shows both are little endian.

Now if I run this multiple time in github pages I get:
```console
js: full_default returned: 0
helpers.js:14 Endianness check: [04 03 02 01]
helpers.js:14 whisper_full_with_state: language pointer: 0x5a4a8c, bytes: [90 70 3d 19]
helpers.js:14 whisper_full_with_state: [danbev] params.language: 𰽙c
helpers.js:14 whisper_full_with_state: [danbev] after copy params.language: 𰽙c
helpers.js:14 whisper_full_with_state: [danbev] not autodetecting....
helpers.js:14 whisper_full_with_state: [danbev] is multilingual params.language: 𰽙c
helpers.js:14 whisper_full_with_state: language pointer: 0x5a4a8c, bytes: [90 70 3d 19]
helpers.js:14 whisper_lang_id: [danbev] lang '𰽙c'
helpers.js:14 whisper_lang_id: unknown language '`p=␙c'
helpers.js:14 [00:00:00.000 --> 00:00:11.000]   So my fellow Americans, ask not what your country can do for you as what you can do for your country.
```
So some form of memory corruption seems to be happening. But perhaps I'm not
looking in the right place. If we look in `examples/whisper.wasm/emscripten.cpp`
we have the following:
```c++
    emscripten::function("full_default", emscripten::optional_override([](size_t index, const emscripten::val & audio, const std::string & lang, int nthreads, bool translate) {
    ...
        printf("%s: [danbev] lang from emscripten: %s\n", __func__, lang.c_str());
        params.language         = whisper_is_multilingual(g_contexts[index]) ? lang.c_str() : "en";
```
So the issue seems to be that the string is passed in from JavaScript like this:
```javascript
var ret = Module.full_default(instance, audio, document.getElementById('language').value, nthreads, translate);
```
And the emscripten function is taking a reference to this string, and then
taking a c pointer to that memory location which is then passed to a thread.
But this memory may be reused which is causing the corruption. Instead if we
make a copy of the string this should work:
```c++
        params.language = whisper_is_multilingual(g_contexts[index]) ? strdup(lang.c_str()) : "en";
```
This output the following now on github pages:
```console
js: full_default returned: 0
helpers.js:14 Endianness check: [04 03 02 01]
helpers.js:14 whisper_full_with_state: language pointer: 0x134396a8, bytes: [65 6e 00 00]
helpers.js:14 whisper_full_with_state: [danbev] params.language: en
helpers.js:14 whisper_full_with_state: [danbev] not autodetecting....
helpers.js:14 whisper_full_with_state: [danbev] is multilingual params.language: en
helpers.js:14 whisper_full_with_state: language pointer: 0x134396a8, bytes: [65 6e 00 00]
helpers.js:14 whisper_lang_id: [danbev] lang 'en'
helpers.js:14 [00:00:00.000 --> 00:00:11.000]   And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

So if we make a copy, and also make sure to free this memory after use this
should work.
```c++
    if (params.language != nullptr && strcmp(params.language, "en") != 0) {
        free((void*)params.language);
    }
```

So to summarize:
When passing a string from javascript to a webassembly module which is what is
happening here means that the memory for the string will be created in
WebAssembly memory, and then passed by reference to the full_default function.
This function will in turn will take a c-string pointer to this memory and pass
it along to another thread (as the params.language pointer). After this the
javascript will continue processing:
```javascript
646                     setTimeout(function() {
647                         var ret = Module.full_default(instance, audio, document.getElementById('language').value, nthreads, transla    te);
648                         console.log('js: full_default returned: ' + ret);
649                         if (ret) {
650                             printTextarea("js: whisper returned: " + ret);
651                         }
652                     }, 100);
```
Which means that the variable will go out of scope and can be reused by the
system. When/if this happens the params.language pointer could be pointing some
memory that now contains different data.

_work in progress_

----

