## ChatML
This was introduced by OpenAI in 2023 and was created to prevent prompt
injection attacks. This is markup language that is used to for OpenAI's API.

```chatml
[
  {
    "role": "system",
    "content": "You are the helpful curator of a virtual art museum."
  },
  {
    "role": "user",
    "content": "Can you show me the famous paintings here?"
  },
  {
    "role": "assistant",
    "content": "Sure, we have 'Starry Night' by Vincent van Gogh, 'Mona Lisa' by Leonardo da Vinci and 'The Persistence of Memory' by Salvador Dali."
  },
  {
    "role": "user",
    "content": "Can you tell me more about 'Starry Night'?"
  },
  {
    "role": "assistant",
    "content": "'Starry Night' is an oil painting by Vincent van Gogh which depicts the view from his asylum room in Saint-Paul-de-Mausole in southern France at night. It's one of his most famous works and is known for its swirling, dreamlike composition and vivid colors."
  },
  {
    "role": "user",
    "content": "Thank you, can you also show me the sculpture section?"
  }
]
```
The idea is that with these roles, system, user, and assistant, the system can
understand the context of the conversation and respond accordingly. So if the
user content contains instructions to perform prompt injection these can be
rejected as those should not be allowed by the user role, those should only be
allowed by the system role. 

This can also be produced as a raw string format instead of json:
```
<|im_start|>system
You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09-01
Current date: 2023-03-01<|im_end|>
<|im_start|>user
How are you<|im_end|>
<|im_start|>assistant
I am doing well!<|im_end|>
<|im_start|>user
How are you now?<|im_end|>
```
This is what llama.cpp uses.
