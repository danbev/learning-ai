###

### Install
```console
$ curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
```

### Configuration for llama-server
Add the following variales to `~/.config/goose/config.yaml`:
```console
OPENAI_BASE_PATH: v1/chat/completions
OPENAI_HOST: http://localhost:8080
```
This can also be done using goose configure and selecting the OpenAI provider,
and then specifying the host url and base path as shown above.

Then we can start goose using:
```console
$ goose
starting session | provider: openai model: Devstral-Small-2507
    logging to /home/danbev/.local/share/goose/sessions/20250723_045837.jsonl
    working directory: /home/danbev/work/ai/learning-ai

Goose is running! Enter your instructions, or try asking what goose can do.

Context: ○○○○○○○○○○ 0% (0/128000 tokens)
( O)> Press Enter to send, Ctrl-J for new line

```
