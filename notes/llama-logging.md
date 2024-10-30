## logging in llama.cpp
For the examples in llama.cpp, in particular main.cpp the logging is initialized
by calling `common_init` in common/common.h:
```c++
// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
void common_init();
```

```c++
void common_init() {
    llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
        if (LOG_DEFAULT_LLAMA <= common_log_verbosity_thold) {
            common_log_add(common_log_main(), level, "%s", text);
        }
    }, NULL);

#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif

    LOG_INF("build: %d (%s) with %s for %s%s\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT, LLAMA_COMPILER, LLAMA_BUILD_TARGET, build_type);
}
```
The function `llama_log_set` is defined in llama.cpp and it takes a callback
function declared in ggml/include/ggml.h:
```
typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
```
So this is a void function which takes a log level and a text string and a void
pointer to some user data.
```c++
void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    ggml_log_set(log_callback, user_data);
    g_logger_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}
```
`ggml_log_set` is setting the logger callback for the ggml library:
```c
void ggml_log_set(ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}
```
And llama.cpp also as a global state variable for the logger callback and the
user data which is set.
So what `common_init` is doing is that it is setting the callback function to
a lambda and the NULL argument is the user data. The body of the function looks
like this:
```c++
        if (LOG_DEFAULT_LLAMA <= common_log_verbosity_thold) {
            common_log_add(common_log_main(), level, "%s", text);
        }
```
The verbosity is an argument or it can be an environment variable which can be
specified as `-lv` or `--verbosity` or `--log-verbosity` and messages with a
higher verbosity then this threshold will be ignored.

So if we set this to 3 then a log message of level 1 (INFO) should be ignored
if I'm understanding this correctly.

`LOG_DEFAULT_LLAMA` is a macro defined in common/log.h:
```c++
#define LOG_DEFAULT_LLAMA 0
```

In `src/llama-impl.h` we can find the macros that are used in llama.cpp:
```c++
//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
void llama_log_internal        (ggml_log_level level, const char * format, ...);
void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG(...)       llama_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) llama_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLAMA_LOG_CONT(...)  llama_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)
```
And in llama.cpp we can find the implementation of `llama_log_internal`:
```c++
void llama_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

static void llama_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = new char[len + 1];
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}
```
Notice that `g_logger_state.log_callback` is used and this is the callback that
we set previously.



### Logging macros
There are a number of helper macros in `common/log.h` which looks like this:
```c++
#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  0,                 __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  0,                 __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, 0,                 __VA_ARGS__)
#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_DEFAULT_DEBUG, __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  0,                 __VA_ARGS__)

#define LOG_INFV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, verbosity, __VA_ARGS__)
#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)
#define LOG_CNTV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  verbosity, __VA_ARGS__)

#define LOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= common_log_verbosity_thold) { \
            common_log_add(common_log_main(), (level), __VA_ARGS__); \
        } \
    } while (0)
```
Notice how this macro uses the verbosity value to determine if the log message
should be added to the logger.


There is an example, [logging.cpp](./fundamentals/llama.cpp/src/logging.cpp) 
that shows how to set the verbosity level using the
`common_log_verbosity_thold` and then using different log macros to see how
they work. There is a pre-processor make target/recipe that can be used to
inspect the expanded macros:
```console
$ cd fundamentals/llama.cpp
$ make logging-pre
...
   do { if ((0) <= common_log_verbosity_thold) { common_log_add(common_log_main(), (GGML_LOG_LEVEL_INFO), "%s: BEVE info\n", __func__); } } while (0);
    do { if ((0) <= common_log_verbosity_thold) { common_log_add(common_log_main(), (GGML_LOG_LEVEL_ERROR), "%s: BEVE error\n", __func__); } } while (0);


    do { if ((3) <= common_log_verbosity_thold) { common_log_add(common_log_main(), (GGML_LOG_LEVEL_INFO), "%s: BEVE v info\n", __func__); } } while (0);
    do { if ((3) <= common_log_verbosity_thold) { common_log_add(common_log_main(), (GGML_LOG_LEVEL_ERROR), "%s: BEVE v error\n", __func__); } } while (0);

    return 0;
}
```
The value of `common_log_verbosity_thold` can be set as a command line argument
for llama-cli and other examples in llama.cpp. These are the logging options:
```console
$ ./llama-cli --help
...

-ld,   --logdir LOGDIR                  path under which to save YAML logs (no logging if unset)
--log-disable                           Log disable
--log-file FNAME                        Log to file
--log-colors                            Enable colored logging
                                        (env: LLAMA_LOG_COLORS)
-v,    --verbose, --log-verbose         Set verbosity level to infinity (i.e. log all messages, useful for
                                        debugging)
-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold. Messages with a higher verbosity will be
                                        ignored.
                                        (env: LLAMA_LOG_VERBOSITY)
--log-prefix                            Enable prefx in log messages
                                        (env: LLAMA_LOG_PREFIX)
--log-timestamps                        Enable timestamps in log messages
                                        (env: LLAMA_LOG_TIMESTAMPS)
```
So if I'm reading this correctly setting a verbosity level of 3 should ignore
all calls with a verbosity greater than 3. This indeed works but I was thinking
that it would be possible to specify that only log message of ERROR should be
displayed, or that if I specify that the log level should be 0
(GGML_LOG_LEVEL_NONE) then no log message should be printed.

### Log levels
The log levels are defined in ggml/include/ggml.h:
```c
    enum ggml_log_level {
        GGML_LOG_LEVEL_NONE  = 0,
        GGML_LOG_LEVEL_INFO  = 1,
        GGML_LOG_LEVEL_WARN  = 2,
        GGML_LOG_LEVEL_ERROR = 3,
        GGML_LOG_LEVEL_DEBUG = 4,
        GGML_LOG_LEVEL_CONT  = 5, // continue previous log
    };
```
