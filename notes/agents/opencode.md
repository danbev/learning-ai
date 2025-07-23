## OpenCode
OpenCode is an opensource terminal user interface for AI code assistants.

github: https://github.com/sst/opencode

### Install
```console
$ npm i -g opencode-ai@latest
```

### MCP Server Configuration
Configure whisper mcp server by adding or updating a `opencode.json` file
with the `mcp` element:
```console
$ cat opencode.json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "whisper": {
      "type": "local",
      "command": [
        "/home/danbev/work/ai/whisper-work/build/bin/whisper-mcp-server",
        "--model",
        "/home/danbev/work/ai/whisper-work/models/ggml-base.en.bin"
      ],
      "enabled": true
    }
  }
}
```
Then restart opencode if it is running.

Example of transcribing an audio file:
```console
Can you transcribe the file audio/samples/jfk.wav?
danbev (04:51 PM)

I'll transcribe the audio file for you using the whisper transcription tool.
claude-sonnet-4-20250514 (04:51 PM)

Whisper_transcribe audio/samples/jfk.wav

 And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

The transcription is: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your
country."
claude-sonnet-4-20250514 (04:51 PM)
```
