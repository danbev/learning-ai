kwargs = {"name": "Alice", "age": 30, "city": "New York", "occupation": "Engineer"}

hub_kwargs_names = ["name", "age"]

hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
#            [                     ] [                          ] [                ]
#             what to store            what to iterate over         filter condition

print("hub_kwargs:", hub_kwargs)

# This is equivalent to the following code:
kwargs = {"name": "Alice", "age": 30, "city": "New York", "occupation": "Engineer"}
hub_kwargs = {}
for name in hub_kwargs_names:
    if name in kwargs:
        hub_kwargs[name] = kwargs.pop(name)

print("hub_kwargs (second method):", hub_kwargs)
