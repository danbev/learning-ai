## Infill
Fill-in-the-middle (FIM) or infill is a special type of prompt that is supported
by CodeLlama (and other perhaps?).
This is a form of code completion to fill in the middle of two code blocks.

For example:
```python
def greeting(name):

   <to be completed, or infilled>

   return result
```
To have this happend it must be passed to the model with a specific prompt, the
same that the model was trained on for this task.
```
<PRE> {prefix} <SUF>{suffix} <MID>
```
The above example would be passed to the model as:
```
<PRE> def greeting(name): <SUF> return result <MID>
```
Where `<PRE>` is the prefix, `<SUF>` is the suffix the first block and the
second block, where we want the completion to be done. `<MID>` indicates that
the model should start predicting after/before this token, it seems to be like
the end of the suffix code block.
