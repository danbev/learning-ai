## Medusa
This is a type of multi-token prediction but the underlaying model is not trained
for this. Instead this is slapped ontop an exising model. So the idea is to take
an existing model which is frozen and train small additional heads on top.

I came accros this when reading about MTP and just wanted to understand what this
is. The MTP heads are trained jointly with the main model.
