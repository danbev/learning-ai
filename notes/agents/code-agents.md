## Code Agents
This document contains information about various code agents and will have
links to more detailed documentation for agents that seem interesting. I don't
used an IDE so those that are VS Code extensions will probably looked at.

### Cline
Is a VS Code extension.

### Roo Code
Is a VS Code extension.

### Aider
github: https://github.com/Aider-AI/aider

Is a terminal based code assistant.

### OpenHands
github: https://github.com/All-Hands-AI/OpenHands/

### goose

### OpenCode
github: https://github.com/sst/opencode

Install:
```console
$ npm i -g opencode-ai@latest
```

Configure whisper mcp server by adding the following to opencode.json (create
it if needed):
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

### Gemini CLI
Command line but does not support other LLMs than Gemini.

### LLXPRT
This is a fork of Gemini CLI that supports multiple LLMs.
```console
$ npm install -g @vybestack/llxprt-code
```

Run using:
```console
$ llxprt
```

