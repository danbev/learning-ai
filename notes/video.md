### Video processing notes

### mtmd ffmpeg

```c++
mtmd_helper_video * mtmd_helper_video_init_from_buf(
        const mtmd_context * mctx,
        const unsigned char * buf, size_t len,
        mtmd_helper_video_init_params params) {
#ifdef MTMD_VIDEO
    auto * ctx = new mtmd_helper_video();

    ctx->mctx                  = mctx;
    ctx->input_buf.assign(buf, buf + len);
    ctx->ffmpeg_bin            = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    ctx->ffprobe_bin           = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    ctx->timestamp_interval_ms = params.timestamp_interval_ms;
```
Then we have:
```c++
    if (!ctx->probe(params.fps_target)) {
        LOG_ERR("%s: ffprobe failed on buffer (is ffprobe in PATH?)\n", __func__);
        delete ctx;
        return nullptr;
    }
```
```c++
    bool probe(float fps_target_arg) {
        const char * input_arg = is_buf_input() ? "pipe:0" : path.c_str();
        const char * cmd[] = {
            ffprobe_bin.c_str(),
            "-v", "quiet",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-select_streams", "v:0",
            "-of", "default=noprint_wrappers=1",
            input_arg,
            nullptr,
        };
```
```console
(gdb) p fps_target_arg 
$14 = 4
(gdb) p input_arg
$15 = 0x7ffff6bb7415 "pipe:0"

(gdb) p cmd
$16 = {0x555555643550 "ffprobe", 0x7ffff6bb741c "-v", 0x7ffff6bb741f "quiet", 0x7ffff6bb7425 "-show_entries",
  0x7ffff6bb7438 "stream=width,height,r_frame_rate,nb_frames,duration", 0x7ffff6bb746c "-select_streams",
  0x7ffff6bb747c "v:0", 0x7ffff6bb7480 "-of", 0x7ffff6bb7484 "default=noprint_wrappers=1", 0x7ffff6bb7415 "pipe:0",
  0x0}
```
So this will be probing the test_video.mp4 file in this case (but it will use
pipe:0 (stdin) instead to for the buffer contents, but we can use the to see
the output:
```console
(gdb) shell ffprobe -v quiet -show_entries stream=width,height,r_frame_rate,nb_frames,duration -select_streams v:0 -of default=noprint_wrappers=1 test_video.mp4
width=1920
height=1080
r_frame_rate=30/1
duration=5.000000
nb_frames=150
```

Next a subprocess_handle is created:
```c++
        subprocess_handle probe_sp;
```
This wraps a [subprocess_s](https://github.com/sheredom/subprocess.h)
Then 
```c++
        if (subprocess_create(cmd,
                subprocess_option_search_user_path | subprocess_option_inherit_environment,
                &probe_sp.proc) != 0) {
            LOG_ERR("%s: failed to launch ffprobe\n", __func__);
            return false;
        }
        probe_sp.created = true;
        probe_sp.alive   = true;
```
```console
(gdb) shell ps aux | grep ffprobe
danbev    546437  0.9  0.1 2402388 46952 pts/0   SLl  13:58   0:01 ffprobe -v quiet -show_entries stream=width,height

(gdb) shell cat /proc/546437/status | grep PPid
PPid:	540166

(gdb) p (int)getpid()
$17 = 540166
```
Next we have:
```c++
        if (is_buf_input()) {
            probe_sp.start_feeder(input_buf);
        }
```
So at this point we have a subprocess which is running ffprope, it is using pipe:0
as its stdin.

When we call start_feeder this will start a thread:
```c++
        void start_feeder(const std::vector<uint8_t> & buf) {
            feeder = std::thread([this, &buf]() {
#ifndef _WIN32
                sigset_t sigpipe_set;  // empty uninitialized bitmask 
                sigemptyset(&sigpipe_set);  // zero all bits (so no signal is set)
                sigaddset(&sigpipe_set, SIGPIPE); // set SIGPIPE bit
                pthread_sigmask(SIG_BLOCK, &sigpipe_set, nullptr); // block all signals in the set for this thread.
                // the above is per threads so only affects the feeder thread.
#endif
                FILE * f = subprocess_stdin(&proc);
                if (!f) {
                    return;
                }
#ifdef F_SETNOSIGPIPE
                fcntl(fileno(f), F_SETNOSIGPIPE, 1); // macos/bsd send it to the process, so turn it off per fd
#endif
                fwrite(buf.data(), 1, buf.size(), f);
                fclose(f);
                proc.stdin_file = nullptr; // prevent double-close in subprocess_destroy
            });
        }
```
The thread will be created and the handle stored in feeder and this function
will return. When the thread is created it will run the lambda.
This will first get a file descriptor for the ffprobe subprocess. Now we have to
keep in mind that a pipe has a kernel buffer which is typlically 64KB on linux)
and fwrite fills that buffer, then blocks waiting for the reader (ffprobe) to
consume it and make room.
```console
feeder_thread          kernel pipe buf          ffprobe
  fwrite()   ---------> [fills buffer] -------> reads what it needs
  blocks ...            [empties]               exists, it is already done
                                                as it just reads the mp4 headers.
  fwrite()   ---------> EPIPE
```
So the next fwrite() will fail with a EPIPE and that could crash the entire
process if not handled. But since we have blocked this signal we will just close
the file descriptor. And since we have closed the file descriptor we set it to
nullptr so that later in it is not closed again:
```c++
int subprocess_destroy(struct subprocess_s *const process) {
  if (process->stdin_file) {
    fclose(process->stdin_file);
    process->stdin_file = SUBPROCESS_NULL;
  }
```
Next we have:
```c++
        uint32_t width  = 0;
        uint32_t height = 0;
        float orig_fps = 0.0f;
        float duration = -1.0f;
        int32_t n_frames_orig = -1;
        char line[256];
        FILE * fp = probe_sp.stdout_pipe();

```
So this is getting a file descriptor to the output of ffprobe which recall was:
```console
width=1920
height=1080
r_frame_rate=30/1
duration=5.000000
nb_frames=150
```
This is then parsed which I'm not showing, and after that we call:
```c++
        probe_sp.stop();
```
```c++
        void stop() {
            // note: alive becomes false on stdout EOF, but the process still needs cleanup
            if (!created) {
                return;
            }
            subprocess_terminate(&proc);

            // join before destroy: feeder holds a FILE* from subprocess_stdin;
            // subprocess_destroy closes it, so the thread must finish first
            if (feeder.joinable()) {
                feeder.join();
            }
            subprocess_join(&proc, nullptr); // reap the child, or else it stays a zombie
            subprocess_destroy(&proc);
            created = false;
            alive   = false;
        }
```
```c++
        fps_target = fps_target_arg > 0.0f ? fps_target_arg : orig_fps;
```
```console
(gdb) p fps_target_arg
$18 = 4
(gdb) p orig_fps
$19 = 30
```
My first thought is why don't we always use the files frames per second as we
have read that from the input. But that for produce 150 frames for a 5 second
video at 30fps. We have to keep in mind that each frame becomes a large number
of tokens. So we extract fewer frames spread evenly accross the video. In our
case we will extract 4 frames per second, so 5 seconds will give us 20 frames
instead of 150.

Next we have:
```c++
    if (!ctx->start_ffmpeg(0.0f)) {
        LOG_ERR("%s: failed to start ffmpeg on buffer (is ffmpeg in PATH?)\n", __func__);
        delete ctx;
        return nullptr;
    }
```
This is pretty similar to ffprobe but runs a different command and also does
not access the output. But there is an important differenct here. The feeder
thread will write to the pipe and fill it. It will then block until the consumer
which now will be the parent llama process reads from it. This allows a lazy
way of reading.

Back in mtmd_helper_bitmap_init_from_buf we then have:
```c++
        mtmd_helper_video_set_id(video_ctx, id); // propagate the hash to the frames

        result = mtmd_bitmap_init_lazy(ctx,
            id.empty() ? nullptr : id.c_str(), // id
            video_ctx,                         // user data
            // callback:
            [](size_t, void * user_data, mtmd_bitmap ** out_bitmap, char ** out_text) -> int {
                auto * vctx = static_cast<mtmd_helper_video *>(user_data);
                char * text = nullptr;
                int ret = mtmd_helper_video_read_next(vctx, out_bitmap, &text);
                *out_text = text; // heap-allocated by read_next; freed automatically by mtmd
                return ret;
            });
         return {result, video_ctx};

```
```c++
MTMD_API mtmd_bitmap * mtmd_bitmap_init_lazy(const mtmd_context * ctx,
                                             const char * id, // usually set to file hash
                                             void * user_data,
                                             mtmd_bitmap_lazy_callback callback);

typedef int(* mtmd_bitmap_lazy_callback)(
    size_t chunk_idx,
    void * user_data,
    mtmd_bitmap ** out_bitmap,
    char ** out_text);
```

Later when mtmd_tokenize_from_parts is called, the mtmd_tokenizer's constructor
will be called:
```c++
int32_t mtmd_tokenize_from_parts(const mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_part * const * parts,
            size_t n_parts,
            bool add_special) {
    for (size_t i = 0; i < n_parts; i++) {
        if ((parts[i]->text == nullptr) == (parts[i]->bitmap == nullptr)) {
            LOG_ERR("%s: part %zu must have either text or bitmap set, not both\n", __func__, i);
            return 1;
        }
        if (parts[i]->text != nullptr && parts[i]->text->text == nullptr) {
            LOG_ERR("%s: part %zu has null text pointer\n", __func__, i);
            return 1;
        }
    }

    try {
--->    mtmd_tokenizer tokenizer(ctx, parts, n_parts, add_special);
        return tokenizer.tokenize(output);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 2;
    }
}
```
```c++
    mtmd_tokenizer(const mtmd_context * ctx,
            const mtmd_input_part * const * input_parts,
            size_t n_parts,
            bool add_special) : ctx(ctx) {
        this->add_special = add_special;
        parse_special = true; // only used for text returned by lazy bitmaps
        vocab         = ctx->vocab;

        for (size_t i = 0; i < n_parts; i++) {
            const mtmd_input_part * p = input_parts[i];
            if (p->text != nullptr) {
                parts.push_back({std::string(p->text->text, p->text->text_len), nullptr, p->text->parse_special});
            } else {
                parts.push_back({"", p->bitmap});
            }
        }

        expand_lazy_bitmaps();
    }
```
When we entry this following function parts will look like this:
```console
(gdb) p parts
$47 = std::vector of length 3, capacity 4 = {
{
    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
    bitmap = 0x0, parse_special = true
},
{
    text = "", bitmap = 0x555557b87da0, parse_special = false
},
{
    text = "Describe the image in detail.<|im_end|>\n<|im_start|>assistant\n", bitmap = 0x0, parse_special = true}}

(gdb) p parts[1].bitmap.lazy_callback 
$49 = (mtmd_bitmap_lazy_callback) 0x7ffff6a87969 <_FUN(size_t, void*, mtmd_bitmap**, char**)>
```
So we have text, a bitmap, and then text. And what we are expanding is the
bitmap below:
```c++
    void expand_lazy_bitmaps() {
        std::vector<part> expanded;
        expanded.reserve(parts.size());

        for (auto & p : parts) {
            // if we have a callback
            if (p.bitmap != nullptr && p.bitmap->lazy_callback) {
                LOG_DBG("%s: expanding lazy bitmap\n", __func__);

                // iterates until all frames are read
                for (size_t i = 0;; i++) {
                    // out parameters for the callback
                    char * out_str = nullptr;
                    mtmd_bitmap * out_bm = nullptr;

                    // this will call mtmd_helper_video_read_next
                    int res = p.bitmap->lazy_callback(i,
                                    p.bitmap->lazy_user_data,
                                    &out_bm,
                                    &out_str);

                    if (out_bm && out_str) {
                        throw std::runtime_error(string_format("lazy callback cannot return both bitmap and text"));
                    }

                    if (res == 0) {
                        // OK, append the returned chunk; lazy part is not yet added
                        if (out_bm) {
                            auto & ptr = bm_from_lazy.emplace_back(out_bm); // remember to free it later
                            expanded.push_back({"", ptr.ptr.get()});
                            LOG_DBG("%s: lazy callback returned bitmap with dimensions %d x %d\n", __func__, out_bm->nx, out_bm->ny);
                        } else if (out_str) {
                            auto & ptr = text_from_lazy.emplace_back(out_str); // remember to free it later
                            expanded.push_back({ptr, nullptr, parse_special});
                            LOG_DBG("%s: lazy callback returned text: %s\n", __func__, out_str);
                        }
                    } else if (res == -1) {
                        // EOF: lazy part removes itself (not added to expanded)
                        break;
                    } else if (res == -2) {
                        // error
                        throw std::runtime_error(string_format("lazy callback returned error"));
                    }
                }
            } else {
                expanded.push_back(std::move(p)); // text part just added directly, nothing done
            }
        }
        // replace the parts with the expanded vector contents.
        parts = std::move(expanded);
    }
```
So the above will iterate over all the 3 parts, and notice that if the part does
not have a bitmap it is just added directly to the expaned vector.
```c++
                    int res = p.bitmap->lazy_callback(i,
                                    p.bitmap->lazy_user_data,
                                    &out_bm,
                                    &out_str);
```
_wip_



### moov
In the Base Media File Format (BMFF) an .mp4 file is structured as a tree of
binary chunks called boxes/atoms. The `moov` (Movie Box) is the meta data for
the table index, timing information, codec config, and byte offsets to decode
the file. Its a "table of contents" and index. It defines how many tracks exist,
how time maps to samples, and precisely where each frame lives inside mdat.
```console
MP4 File
├── ftyp (File Type Box: brands, compatibility)
├── moov (Movie Box: metadata, indices, codec configurations)
└── mdat (Media Data Box: raw audio/video payload)
```
When recording th moov is naturally written after the actual data as that is the
point where it knows all the information. But when player needs to read the
data it first has to access this information. If it is at the end of the file
will does not want to  have to read the entire file first, but perhaps seek to
a position to find it instead.
