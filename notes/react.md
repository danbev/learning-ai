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
