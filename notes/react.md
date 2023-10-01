## ReAct (Reasoning + Acting)
Allows an LLM to Reason and Act to the world around it.

### Example
The following is an example of a ReAct agent which can be found in
[agent-serper.py](../langchain/src/agent-serper.py). This example uses
[SERPAPI](https://serpapi.com/) to allow the agent to search the web for
information.

The example uses the followig prompt:
```
"Who is Austin Powers? What is his current age raised to the 0.23 power?
Return his age in the result"
```

```console
$ cd langchain/
$ python3 -m venv langch
$ source langch/bin/activate
(langch) $ python src/agent-serper.py 

> Entering new AgentExecutor chain...
Thought: I need to use Google Search to answer the first part of the question and a calculator to answer the second part.

Action:
```
{
  "action": "google_serper",
  "action_input": "Who is Austin Powers?"
}
```


Observation: Sir Austin Danger Powers KBE is a fictional character from the Austin Powers series of films, and is created and portrayed by Mike Myers. He is the protagonist of Austin Powers: International Man of Mystery (1997), The Spy Who Shagged Me (1999) and Austin Powers in Goldmember (2002).
Thought: I need to use a calculator to answer the second part of the question.

Action:
```
{
  "action": "Calculator",
  "action_input": "Austin Powers' current age raised to the 0.23 power"
}
```


Observation: Answer: 2.4590258885237315
Thought: I now know the final answer

Final Answer: Sir Austin Danger Powers KBE is a fictional character from the Austin Powers series of films, and is created and portrayed by Mike Myers. He is the protagonist of Austin Powers: International Man of Mystery (1997), The Spy Who Shagged Me (1999) and Austin Powers in Goldmember (2002). Austin Powers' current age raised to the 0.23 power is 2.4590258885237315.

> Finished chain.
Spent a total of 1959 tokens
Sir Austin Danger Powers KBE is a fictional character from the Austin Powers series of films, and is created and portrayed by Mike Myers. He is the protagonist of Austin Powers: International Man of Mystery (1997), The Spy Who Shagged Me (1999) and Austin Powers in Goldmember (2002). His current age raised to the 0.23 power is 2.4590258885237315.
```

The prompts uses for these types of agents looks like this:
```
System prompt:
Answer the following questions as best you can. You have access to the following tools:

google_serper: A low-cost Google Search API.Useful for when you need to answer questions about current events.Input should be a search query.
Calculator: Useful for when you need to answer questions about math.

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: google_serper, Calculator

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.

User prompt:
{input}

{agent_scratchpad}
```
The `agent_scratchpad` is a place for the agent to store previous throughts
and is how it can know what it has already found out. This is not a context
that is passed to the LLM.

If we look at the system prompt we see a pattern of `Thought:`, `Action:`, and
`Observation:`. This is the ReAct loop. The agent will think about what to do,
do it, and then observe the result. This loop can repeat as many times as needed
or as configured by the `max_iterations` parameter. Once the agent has the final
answer it will print it out and the chain will end.

This is modeled after [Modular Reasoning, Knowledge and Language] (MRKL,
pronouned as "mircle").

[modular reasoning, knowledge and language]: https://arxiv.org/pdf/2205.00445.pdf
