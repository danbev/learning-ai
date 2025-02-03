## Server unit test notes

### test_completion_with_response_format failure
When running this test locally (ubuntu 24.04) it fails with the following error:
```console
(venv) $ ./tests.sh unit/test_chat_completion.py::test_completion_with_response_format -s -v -k "response_format3"
======================================================== test session starts =========================================================
platform linux -- Python 3.11.11, pytest-8.3.4, pluggy-1.5.0 -- /home/danbev/work/ai/llama.cpp-debug/examples/server/tests/venv/bin/python3.11
cachedir: .pytest_cache
rootdir: /home/danbev/work/ai/llama.cpp-debug/examples/server/tests
configfile: pytest.ini
plugins: anyio-4.8.0
collected 7 items / 6 deselected / 1 selected                                                                                        

unit/test_chat_completion.py::test_completion_with_response_format[response_format3-0-None] bench: starting server with: ../../../build/bin/llama-server --host 127.0.0.1 --port 8080 --temp 0.8 --seed 42 --hf-repo ggml-org/models --hf-file tinyllamas/stories260K.gguf --batch-size 32 --alias tinyllama-2 --ctx-size 256 --parallel 2 --n-predict 64
server pid=158649, pytest pid=158648
Waiting for server to start...
ggml_cuda_init: failed to initialize CUDA: unknown error
register_backend: registered backend CUDA (0 devices)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
load_backend: failed to find ggml_backend_init in /home/danbev/work/ai/llama.cpp-debug/build/bin/libggml-cuda.so
load_backend: failed to find ggml_backend_init in /home/danbev/work/ai/llama.cpp-debug/build/bin/libggml-cpu.so
build: 4621 (6eecde3c) with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu (debug)
system info: n_threads = 4, n_threads_batch = 4, total_threads = 16

system_info: n_threads = 4 (n_threads_batch = 4) / 16 | CUDA : ARCHS = 600,610,700,750 | F16 = 1 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1 | LLAMAFILE = 1 | 

main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 15
main: loading model
srv    load_model: loading model '/home/danbev/.cache/llama.cpp/ggml-org_models_tinyllamas_stories260K.gguf'
common_download_file: previous metadata file found /home/danbev/.cache/llama.cpp/ggml-org_models_tinyllamas_stories260K.gguf.json: {"etag":"\"21c86626afafc826a642338679227c24\"","lastModified":"Tue, 20 Feb 2024 09:21:22 GMT","url":"https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"}
curl_perform_with_retry: Trying to download from https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf (attempt 1 of 3)...
request: GET /health 127.0.0.1 503
Response from server {
  "error": {
    "code": 503,
    "message": "Loading model",
    "type": "unavailable_error"
  }
}
Waiting for server to start...
llama_model_loader: loaded meta data with 19 key-value pairs and 48 tensors from /home/danbev/.cache/llama.cpp/ggml-org_models_tinyllamas_stories260K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                      tokenizer.ggml.tokens arr[str,512]     = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv   1:                      tokenizer.ggml.scores arr[f32,512]     = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv   2:                  tokenizer.ggml.token_type arr[i32,512]     = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv   3:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv   4:                       general.architecture str              = llama
llama_model_loader: - kv   5:                               general.name str              = llama
llama_model_loader: - kv   6:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv   7:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv   8:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv   9:          tokenizer.ggml.seperator_token_id u32              = 4294967295
llama_model_loader: - kv  10:            tokenizer.ggml.padding_token_id u32              = 4294967295
llama_model_loader: - kv  11:                       llama.context_length u32              = 128
llama_model_loader: - kv  12:                     llama.embedding_length u32              = 64
llama_model_loader: - kv  13:                  llama.feed_forward_length u32              = 172
llama_model_loader: - kv  14:                 llama.attention.head_count u32              = 8
llama_model_loader: - kv  15:              llama.attention.head_count_kv u32              = 4
llama_model_loader: - kv  16:                          llama.block_count u32              = 5
llama_model_loader: - kv  17:                 llama.rope.dimension_count u32              = 8
llama_model_loader: - kv  18:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - type  f32:   48 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = all F32 (guessed)
print_info: file size   = 1.12 MiB (32.00 BPW) 
load: bad special token: 'tokenizer.ggml.seperator_token_id' = 4294967295, using default id -1
load: bad special token: 'tokenizer.ggml.padding_token_id' = 4294967295, using default id -1
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 3
load: token to piece cache size = 0.0008 MB
print_info: arch             = llama
print_info: vocab_only       = 0
print_info: n_ctx_train      = 128
print_info: n_embd           = 64
print_info: n_layer          = 5
print_info: n_head           = 8
print_info: n_head_kv        = 4
print_info: n_rot            = 8
print_info: n_swa            = 0
print_info: n_embd_head_k    = 8
print_info: n_embd_head_v    = 8
print_info: n_gqa            = 2
print_info: n_embd_k_gqa     = 32
print_info: n_embd_v_gqa     = 32
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: n_ff             = 172
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = linear
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 128
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = ?B
print_info: model params     = 292.80 K
print_info: general.name     = llama
print_info: vocab type       = SPM
print_info: n_vocab          = 512
print_info: n_merges         = 0
print_info: BOS token        = 1 '<s>'
print_info: EOS token        = 2 '</s>'
print_info: UNK token        = 0 '<unk>'
print_info: LF token         = 13 '<0x0A>'
print_info: EOG token        = 2 '</s>'
print_info: max token length = 9
load_tensors:   CPU_Mapped model buffer size =     1.12 MiB
llama_init_from_model: n_batch is less than GGML_KQ_MASK_PAD - increasing to 64
llama_init_from_model: n_seq_max     = 2
llama_init_from_model: n_ctx         = 256
llama_init_from_model: n_ctx_per_seq = 128
llama_init_from_model: n_batch       = 64
llama_init_from_model: n_ubatch      = 64
llama_init_from_model: flash_attn    = 0
llama_init_from_model: freq_base     = 10000.0
llama_init_from_model: freq_scale    = 1
llama_kv_cache_init: kv_size = 256, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 5, can_shift = 1
llama_kv_cache_init:        CPU KV buffer size =     0.16 MiB
llama_init_from_model: KV self size  =    0.16 MiB, K (f16):    0.08 MiB, V (f16):    0.08 MiB
llama_init_from_model:        CPU  output buffer size =     0.00 MiB
llama_init_from_model:        CPU compute buffer size =     0.63 MiB
llama_init_from_model: graph nodes  = 166
llama_init_from_model: graph splits = 1
common_init_from_params: setting dry_penalty_last_n to ctx_size = 256
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
srv          init: initializing slots, n_slots = 2
slot         init: id  0 | task -1 | new slot n_ctx_slot = 128
slot         init: id  1 | task -1 | new slot n_ctx_slot = 128
main: model loaded
main: chat template, chat_template: 
                {%- for message in messages -%}
                    {{- "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n" -}}
                {%- endfor -%}
                {%- if add_generation_prompt -%}
                    {{- "<|im_start|>assistant\n" -}}
                {%- endif -%}
            , example_format: '<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
'
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
request: GET /health 127.0.0.1 200
Response from server {
  "status": "ok"
}
terminate called after throwing an instance of 'std::runtime_error'
  what():  response_format type must be one of "text" or "json_object", but got: sound
FAILEDStopping server with pid=158649


============================================================== FAILURES ==============================================================
___________________________________ test_completion_with_response_format[response_format3-0-None] ____________________________________

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x786fe47b0990>, method = 'POST', url = '/chat/completions'
body = b'{"max_tokens": 0, "messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Write an example"}], "response_format": {"type": "sound"}}'
headers = {'User-Agent': 'python-requests/2.32.3', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Length': '180', 'Content-Type': 'application/json'}
retries = Retry(total=0, connect=None, read=False, redirect=None, status=None), redirect = False, assert_same_host = False
timeout = Timeout(connect=None, read=None, total=None), pool_timeout = None, release_conn = False, chunked = False, body_pos = None
preload_content = False, decode_content = False, response_kw = {}
parsed_url = Url(scheme=None, auth=None, host=None, port=None, path='/chat/completions', query=None, fragment=None)
destination_scheme = None, conn = None, release_this_conn = True, http_tunnel_required = False, err = None, clean_exit = False

    def urlopen(  # type: ignore[override]
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        retries: Retry | bool | int | None = None,
        redirect: bool = True,
        assert_same_host: bool = True,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        pool_timeout: int | None = None,
        release_conn: bool | None = None,
        chunked: bool = False,
        body_pos: _TYPE_BODY_POSITION | None = None,
        preload_content: bool = True,
        decode_content: bool = True,
        **response_kw: typing.Any,
    ) -> BaseHTTPResponse:
        """
        Get a connection from the pool and perform an HTTP request. This is the
        lowest level call for making a request, so you'll need to specify all
        the raw details.
    
        .. note::
    
           More commonly, it's appropriate to use a convenience method
           such as :meth:`request`.
    
        .. note::
    
           `release_conn` will only behave as expected if
           `preload_content=False` because we want to make
           `preload_content=False` the default behaviour someday soon without
           breaking backwards compatibility.
    
        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)
    
        :param url:
            The URL to perform the request on.
    
        :param body:
            Data to send in the request body, either :class:`str`, :class:`bytes`,
            an iterable of :class:`str`/:class:`bytes`, or a file-like object.
    
        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.
    
        :param retries:
            Configure the number of retries to allow before raising a
            :class:`~urllib3.exceptions.MaxRetryError` exception.
    
            If ``None`` (default) will retry 3 times, see ``Retry.DEFAULT``. Pass a
            :class:`~urllib3.util.retry.Retry` object for fine-grained control
            over different types of retries.
            Pass an integer number to retry connection errors that many times,
            but no other types of errors. Pass zero to never retry.
    
            If ``False``, then retries are disabled and any exception is raised
            immediately. Also, instead of raising a MaxRetryError on redirects,
            the redirect response will be returned.
    
        :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.
    
        :param redirect:
            If True, automatically handle redirects (status codes 301, 302,
            303, 307, 308). Each redirect counts as a retry. Disabling retries
            will disable redirect, too.
    
        :param assert_same_host:
            If ``True``, will make sure that the host of the pool requests is
            consistent else will raise HostChangedError. When ``False``, you can
            use the pool on an HTTP proxy and request foreign hosts.
    
        :param timeout:
            If specified, overrides the default timeout for this one
            request. It may be a float (in seconds) or an instance of
            :class:`urllib3.util.Timeout`.
    
        :param pool_timeout:
            If set and the pool is set to block=True, then this method will
            block for ``pool_timeout`` seconds and raise EmptyPoolError if no
            connection is available within the time period.
    
        :param bool preload_content:
            If True, the response's body will be preloaded into memory.
    
        :param bool decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
    
        :param release_conn:
            If False, then the urlopen call will not release the connection
            back into the pool once a response is received (but will release if
            you read the entire contents of the response such as when
            `preload_content=True`). This is useful if you're not preloading
            the response's content immediately. You will need to call
            ``r.release_conn()`` on the response ``r`` to return the connection
            back into the pool. If None, it takes the value of ``preload_content``
            which defaults to ``True``.
    
        :param bool chunked:
            If True, urllib3 will send the body using chunked transfer
            encoding. Otherwise, urllib3 will send the body using the standard
            content-length form. Defaults to False.
    
        :param int body_pos:
            Position to seek to in file-like body in the event of a retry or
            redirect. Typically this won't need to be set because urllib3 will
            auto-populate the value when needed.
        """
        parsed_url = parse_url(url)
        destination_scheme = parsed_url.scheme
    
        if headers is None:
            headers = self.headers
    
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)
    
        if release_conn is None:
            release_conn = preload_content
    
        # Check host
        if assert_same_host and not self.is_same_host(url):
            raise HostChangedError(self, url, retries)
    
        # Ensure that the URL we're connecting to is properly encoded
        if url.startswith("/"):
            url = to_str(_encode_target(url))
        else:
            url = to_str(parsed_url.url)
    
        conn = None
    
        # Track whether `conn` needs to be released before
        # returning/raising/recursing. Update this variable if necessary, and
        # leave `release_conn` constant throughout the function. That way, if
        # the function recurses, the original value of `release_conn` will be
        # passed down into the recursive call, and its value will be respected.
        #
        # See issue #651 [1] for details.
        #
        # [1] <https://github.com/urllib3/urllib3/issues/651>
        release_this_conn = release_conn
    
        http_tunnel_required = connection_requires_http_tunnel(
            self.proxy, self.proxy_config, destination_scheme
        )
    
        # Merge the proxy headers. Only done when not using HTTP CONNECT. We
        # have to copy the headers dict so we can safely change it without those
        # changes being reflected in anyone else's copy.
        if not http_tunnel_required:
            headers = headers.copy()  # type: ignore[attr-defined]
            headers.update(self.proxy_headers)  # type: ignore[union-attr]
    
        # Must keep the exception bound to a separate variable or else Python 3
        # complains about UnboundLocalError.
        err = None
    
        # Keep track of whether we cleanly exited the except block. This
        # ensures we do proper cleanup in finally.
        clean_exit = False
    
        # Rewind body position, if needed. Record current position
        # for future rewinds in the event of a redirect/retry.
        body_pos = set_file_position(body, body_pos)
    
        try:
            # Request a connection from the queue.
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)
    
            conn.timeout = timeout_obj.connect_timeout  # type: ignore[assignment]
    
            # Is this a closed/new connection that requires CONNECT tunnelling?
            if self.proxy is not None and http_tunnel_required and conn.is_closed:
                try:
                    self._prepare_proxy(conn)
                except (BaseSSLError, OSError, SocketTimeout) as e:
                    self._raise_timeout(
                        err=e, url=self.proxy.url, timeout_value=conn.timeout
                    )
                    raise
    
            # If we're going to release the connection in ``finally:``, then
            # the response doesn't need to know about the connection. Otherwise
            # it will also try to release it and we'll have a double-release
            # mess.
            response_conn = conn if not release_conn else None
    
            # Make the request on the HTTPConnection object
>           response = self._make_request(
                conn,
                method,
                url,
                timeout=timeout_obj,
                body=body,
                headers=headers,
                chunked=chunked,
                retries=retries,
                response_conn=response_conn,
                preload_content=preload_content,
                decode_content=decode_content,
                **response_kw,
            )

venv/lib/python3.11/site-packages/urllib3/connectionpool.py:787: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
venv/lib/python3.11/site-packages/urllib3/connectionpool.py:534: in _make_request
    response = conn.getresponse()
venv/lib/python3.11/site-packages/urllib3/connection.py:516: in getresponse
    httplib_response = super().getresponse()
/usr/lib/python3.11/http/client.py:1395: in getresponse
    response.begin()
/usr/lib/python3.11/http/client.py:325: in begin
    version, status, reason = self._read_status()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <http.client.HTTPResponse object at 0x786fe4763610>

    def _read_status(self):
        line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
        if len(line) > _MAXLINE:
            raise LineTooLong("status line")
        if self.debuglevel > 0:
            print("reply:", repr(line))
        if not line:
            # Presumably, the server closed the connection before
            # sending a valid response.
>           raise RemoteDisconnected("Remote end closed connection without"
                                     " response")
E           http.client.RemoteDisconnected: Remote end closed connection without response

/usr/lib/python3.11/http/client.py:294: RemoteDisconnected

During handling of the above exception, another exception occurred:

self = <requests.adapters.HTTPAdapter object at 0x786fe47b0790>, request = <PreparedRequest [POST]>, stream = False
timeout = Timeout(connect=None, read=None, total=None), verify = True, cert = None, proxies = OrderedDict()

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        """Sends PreparedRequest object. Returns Response object.
    
        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple or urllib3 Timeout object
        :param verify: (optional) Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        :rtype: requests.Response
        """
    
        try:
            conn = self.get_connection_with_tls_context(
                request, verify, proxies=proxies, cert=cert
            )
        except LocationValueError as e:
            raise InvalidURL(e, request=request)
    
        self.cert_verify(conn, request.url, verify, cert)
        url = self.request_url(request, proxies)
        self.add_headers(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )
    
        chunked = not (request.body is None or "Content-Length" in request.headers)
    
        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
                timeout = TimeoutSauce(connect=connect, read=read)
            except ValueError:
                raise ValueError(
                    f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "
                    f"or a single float to set both timeouts to the same value."
                )
        elif isinstance(timeout, TimeoutSauce):
            pass
        else:
            timeout = TimeoutSauce(connect=timeout, read=timeout)
    
        try:
>           resp = conn.urlopen(
                method=request.method,
                url=url,
                body=request.body,
                headers=request.headers,
                redirect=False,
                assert_same_host=False,
                preload_content=False,
                decode_content=False,
                retries=self.max_retries,
                timeout=timeout,
                chunked=chunked,
            )

venv/lib/python3.11/site-packages/requests/adapters.py:667: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
venv/lib/python3.11/site-packages/urllib3/connectionpool.py:841: in urlopen
    retries = retries.increment(
venv/lib/python3.11/site-packages/urllib3/util/retry.py:474: in increment
    raise reraise(type(error), error, _stacktrace)
venv/lib/python3.11/site-packages/urllib3/util/util.py:38: in reraise
    raise value.with_traceback(tb)
venv/lib/python3.11/site-packages/urllib3/connectionpool.py:787: in urlopen
    response = self._make_request(
venv/lib/python3.11/site-packages/urllib3/connectionpool.py:534: in _make_request
    response = conn.getresponse()
venv/lib/python3.11/site-packages/urllib3/connection.py:516: in getresponse
    httplib_response = super().getresponse()
/usr/lib/python3.11/http/client.py:1395: in getresponse
    response.begin()
/usr/lib/python3.11/http/client.py:325: in begin
    version, status, reason = self._read_status()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <http.client.HTTPResponse object at 0x786fe4763610>

    def _read_status(self):
        line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
        if len(line) > _MAXLINE:
            raise LineTooLong("status line")
        if self.debuglevel > 0:
            print("reply:", repr(line))
        if not line:
            # Presumably, the server closed the connection before
            # sending a valid response.
>           raise RemoteDisconnected("Remote end closed connection without"
                                     " response")
E           urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))

/usr/lib/python3.11/http/client.py:294: ProtocolError

During handling of the above exception, another exception occurred:

response_format = {'type': 'sound'}, n_predicted = 0, re_content = None

    @pytest.mark.parametrize("response_format,n_predicted,re_content", [
        ({"type": "json_object", "schema": {"const": "42"}}, 6, "\"42\""),
        ({"type": "json_object", "schema": {"items": [{"type": "integer"}]}}, 10, "[ -3000 ]"),
        ({"type": "json_object"}, 10, "(\\{|John)+"),
        ({"type": "sound"}, 0, None),
        # invalid response format (expected to fail)
        ({"type": "json_object", "schema": 123}, 0, None),
        ({"type": "json_object", "schema": {"type": 123}}, 0, None),
        ({"type": "json_object", "schema": {"type": "hiccup"}}, 0, None),
    ])
    def test_completion_with_response_format(response_format: dict, n_predicted: int, re_content: str | None):
        global server
        server.start()
>       res = server.make_request("POST", "/chat/completions", data={
            "max_tokens": n_predicted,
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write an example"},
            ],
            "response_format": response_format,
        })

unit/test_chat_completion.py:152: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
utils.py:238: in make_request
    response = requests.post(url, headers=headers, json=data, timeout=timeout)
venv/lib/python3.11/site-packages/requests/api.py:115: in post
    return request("post", url, data=data, json=json, **kwargs)
venv/lib/python3.11/site-packages/requests/api.py:59: in request
    return session.request(method=method, url=url, **kwargs)
venv/lib/python3.11/site-packages/requests/sessions.py:589: in request
    resp = self.send(prep, **send_kwargs)
venv/lib/python3.11/site-packages/requests/sessions.py:703: in send
    r = adapter.send(request, **kwargs)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <requests.adapters.HTTPAdapter object at 0x786fe47b0790>, request = <PreparedRequest [POST]>, stream = False
timeout = Timeout(connect=None, read=None, total=None), verify = True, cert = None, proxies = OrderedDict()

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        """Sends PreparedRequest object. Returns Response object.
    
        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple or urllib3 Timeout object
        :param verify: (optional) Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        :rtype: requests.Response
        """
    
        try:
            conn = self.get_connection_with_tls_context(
                request, verify, proxies=proxies, cert=cert
            )
        except LocationValueError as e:
            raise InvalidURL(e, request=request)
    
        self.cert_verify(conn, request.url, verify, cert)
        url = self.request_url(request, proxies)
        self.add_headers(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )
    
        chunked = not (request.body is None or "Content-Length" in request.headers)
    
        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
                timeout = TimeoutSauce(connect=connect, read=read)
            except ValueError:
                raise ValueError(
                    f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "
                    f"or a single float to set both timeouts to the same value."
                )
        elif isinstance(timeout, TimeoutSauce):
            pass
        else:
            timeout = TimeoutSauce(connect=timeout, read=timeout)
    
        try:
            resp = conn.urlopen(
                method=request.method,
                url=url,
                body=request.body,
                headers=request.headers,
                redirect=False,
                assert_same_host=False,
                preload_content=False,
                decode_content=False,
                retries=self.max_retries,
                timeout=timeout,
                chunked=chunked,
            )
    
        except (ProtocolError, OSError) as err:
>           raise ConnectionError(err, request=request)
E           requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))

venv/lib/python3.11/site-packages/requests/adapters.py:682: ConnectionError
====================================================== short test summary info =======================================================
FAILED unit/test_chat_completion.py::test_completion_with_response_format[response_format3-0-None] - requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
================================================== 1 failed, 6 deselected in 2.35s ===================================================
```
This error comes from the following test:
```python
@pytest.mark.parametrize("response_format,n_predicted,re_content", [
    ({"type": "json_object", "schema": {"const": "42"}}, 6, "\"42\""),
    ({"type": "json_object", "schema": {"items": [{"type": "integer"}]}}, 10, "[ -3000 ]"),
    ({"type": "json_object"}, 10, "(\\{|John)+"),
    ({"type": "sound"}, 0, None),
    # invalid response format (expected to fail)
    ({"type": "json_object", "schema": 123}, 0, None),
    ({"type": "json_object", "schema": {"type": 123}}, 0, None),
    ({"type": "json_object", "schema": {"type": "hiccup"}}, 0, None),
])
def test_completion_with_response_format(response_format: dict, n_predicted: int, re_content: str | None):
```
We can see that this is a parameterized test and the one we are running is the
one with the invalid response format set to `sound`. This is expected to fail on
the server side and this error originates from server.cpp:
```c++
    const auto handle_chat_completions = [&ctx_server, &params, &res_error, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
    ...
        json data = oaicompat_completion_params_parse(body, params.use_jinja, ctx_server.chat_templates);
```
And in utils.hpp we have `oaicompat_completion_params_parse`:
```c++
static json oaicompat_completion_params_parse(
    const json & body, /* openai api json semantics */
    bool use_jinja,
    const common_chat_templates & chat_templates)
{
    ...

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            llama_params["json_schema"] = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            json json_schema = json_value(response_format, "json_schema", json::object());
            llama_params["json_schema"] = json_value(json_schema, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }
    ...
}
```
So this is throwing a runtime error.
I noticed that in other functions is server.cpp, for example in
`handle_completions_impl` there are try/catch blocks around calls:
```c++
    const auto handle_completions_impl = [&ctx_server, &res_error, &res_ok](
            server_task_type type,
            json & data,
            std::function<bool()> is_connection_closed,
            httplib::Response & res,
            oaicompat_type oaicompat) {
            ...
        try {
            const auto & prompt = data.at("prompt");
            LOG_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(type);

                task.id    = ctx_server.queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(tokenized_prompts[i]);
                task.params           = server_task::params_from_json_cmpl(
                                            ctx_server.ctx,
                                            ctx_server.params_base,
                                            data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat                 = oaicompat;
                task.params.oaicompat_cmpl_id         = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(task);
            }
        } catch (const std::exception & e) {
            res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return;
        }
```
Perhaps there should be a try-catch block around the call to
`oaicompat_completion_params_parse`. Adding a simliar block allows the test to
pass.
```console
diff --git a/examples/server/server.cpp b/examples/server/server.cpp
index e0acc470..a130b891 100644
--- a/examples/server/server.cpp
+++ b/examples/server/server.cpp
@@ -4048,7 +4048,12 @@ int main(int argc, char ** argv) {
         }

         auto body = json::parse(req.body);
-        json data = oaicompat_completion_params_parse(body, params.use_jinja, ctx_server.chat_templates);
+        json data;
+        try {
+            data = oaicompat_completion_params_parse(body, params.use_jinja, ctx_server.chat_templates);
+        } catch (const std::exception & e) {
+            res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
+        }

         return handle_completions_impl(
             SERVER_TASK_TYPE_COMPLETION,

```
But the strange thing is that this test passes on the [CI](https://github.com/ggerganov/llama.cpp/actions/runs/13107737340/job/36565269529#step:11:44) server.

I'm using ubuntu:
```console
$ lsb_release -a
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 24.04.1 LTS
Release:	24.04
Codename:	noble
```
And the CI server is using:
```console
Operating System
  Ubuntu
  24.04.1
  LTS
```
I'm using python 3.11 (was using 3.12 but switched to 3.11 to see if that was
the issue) and the CI server:
```console
platform linux -- Python 3.11.11, pytest-8.3.4, pluggy-1.5.0 -- /opt/hostedtoolcache/Python/3.11.11/x64/bin/python
cachedir: .pytest_cache
rootdir: /home/runner/work/llama.cpp/llama.cpp/examples/server/tests
configfile: pytest.ini
```
But still, what I'm seeing is that the error is being thrown in server but
not getting handled. How can this differ between the two systems?  

Looking closer at the logs from the test I noticed:
```console
>           raise RemoteDisconnected("Remote end closed connection without"
                                     " response")
E           http.client.RemoteDisconnected: Remote end closed connection without response

/usr/lib/python3.11/http/client.py:294: RemoteDisconnected

During handling of the above exception, another exception occurred:
```
Could it be that when running locally the server is to fast in closing the
connection/terminating the process and the client is not able to handle the
actual error returned as the connection handler in python is getting a
connection error before.

Looking closer at the routing in cpp_httplib I noticed that the routing call
is coming from:
```c++
  // Routing
  auto routed = false;
#ifdef CPPHTTPLIB_NO_EXCEPTIONS
  routed = routing(req, res, strm);
#else
  try {
    routed = routing(req, res, strm);
  } catch (std::exception &e) {
    if (exception_handler_) {
      auto ep = std::current_exception();
      exception_handler_(req, res, ep);
      routed = true;
    } else {
      res.status = StatusCode::InternalServerError_500;
      std::string val;
      auto s = e.what();
      for (size_t i = 0; s[i]; i++) {
        switch (s[i]) {
        case '\r': val += "\\r"; break;
        case '\n': val += "\\n"; break;
        default: val += s[i]; break;
        }
      }
      res.set_header("EXCEPTION_WHAT", val);
    }
  } catch (...) {
    if (exception_handler_) {
      auto ep = std::current_exception();
      exception_handler_(req, res, ep);
      routed = true;
    } else {
      res.status = StatusCode::InternalServerError_500;
      res.set_header("EXCEPTION_WHAT", "UNKNOWN");
    }
  }
#endif
```
And this is set in debug mode in utils.hpp:
```c++
#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
```


