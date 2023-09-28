## Langchain
Langchain is a library for creating AI/ML applications.

There are a few examples in the [langchain](../langchain) directory.

### Agents
Agents are like tools for the LLM enabling an LLM to get information from the
outside world, like searching the web, querying a database, or performing math
calculations.

```
Agents use an LLM to determine which actions to take and in what order. An
action can either be using a tool and observing its output, or returning to the
user.
```

In a "normal" Langchain chain we specify the actions to be taken. For example
we in [vec_cve.py](../langchain/src/vex_cve.py) we use chains. 

But with agents we the LLM decides what action that is to be taken. 

The are two types of agents:
* Action agents
* Plan and execute agents
