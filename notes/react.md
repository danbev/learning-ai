## ReAct (Reasoning + Acting)
Allows an LLM to Reason and Act to the world around it.

### Example
The following is an example of a ReAct agent which can be found in
[agent-serper.py](../langchain/src/agent-serper.py). This example uses
[SERPAPI](https://serpapi.com/) to allow the agent to search the web for
information.

```console
$ cd langchain/
$ python3 -m venv langch
$ source langch/bin/activate
(langch) $ python src/agent-serper.py 


> Entering new AgentExecutor chain...
 I need to find out who Austin Powers is and then use a calculator to calculate
his age raised to the 0.23 power
Action: google_serper
Action Input: "Who is Austin Powers"
Observation: Sir Austin Danger Powers KBE is a fictional character from the Austin Powers series of films, and is created and portrayed by Mike Myers. He is the protagonist of Austin Powers: International Man of Mystery (1997), The Spy Who Shagged Me (1999) and Austin Powers in Goldmember (2002).

Thought: I need to find out Austin Powers' age
Action: google_serper
Action Input: "What is Austin Powers' age"
Observation: Austin Powers is a series of American satirical spy comedy films created by Mike Myers, who also stars as the eponymous title character as well as his ... Sir Austin Danger Powers is the main protagonist of the Austin Powers franchise. He is a British secret agent, Dr. Evil's twin brother, and Mini-Me's Uncle. He ... Sir Austin Danger Powers KBE is a fictional character from the Austin Powers series of films, and is created and portrayed by Mike Myers. If we can assume that Austin Powers is supposed to be approximately the same age as Mike Myers (the actor who portrays him), then Austin Powers ... PG-13 for sexual themes/content and language ... There is some sexual references and a scene where Austin Powers and another character have sex in ... Because it's a comedy, the rating system gives it a PG-13, but the material would clearly get an R if it appeared in a drama. Do not kid… A world-class playboy and part-time secret agent from the 1960s emerges after thirty years in a cryogenic state to battle with his nemesis Dr. Evil.A world- ... The creation of comedian Mike Myers, Austin Powers is the goofy hero of the 1997 film Austin Powers: International Man of Mystery. The Austin Powers ... The comedy, about a 1960s British spy with a libido as oversized as his cravate, turns 25 on May 2. It follows the title character, a spy ( ... Missing: age | Show results with:age. Austin Powers (character). Austin Powers (character). Fame Meter (2/100) Has their own Wikipedia Page. Austin Powers (character). Age. 83 years old. Family ...

Thought: I now have the age of Austin Powers
Action: Calculator
Action Input: 83^0.23
Observation: Answer: 2.763045020279681

Thought: I now know the final answer
Final Answer: Austin Powers is 83 years old and his current age raised to the 0.23 power is 2.763045020279681.

> Finished chain.
result='Austin Powers is 83 years old and his current age raised to the 0.23 power is 2.763045020279681.
```
