import pandas as pd

data = pd.DataFrame()
print(data)
print(data.count())
print('\n'.join(dir(data)))

print(data.size)

l = {
        "Name": ["Fletch", "Mr. Poon", "Mr. Sinilinden"],
        "Age": [48, 47, 49],
}
people = pd.DataFrame(l)
print(people)
print(people["Name"][:0])


names = pd.Series(["Fletch", "Mr. Poon", "Mr. Sinilinden"])
ages = pd.Series([48, 47, 49])
people2 = pd.DataFrame({"Name": names, "Age": ages})
print(people2)

data = [
        ["Fletch", 48],
        ["Mr.Pool", 47],
        ["Mr.Sinilinden", 49],
]

people3 = pd.DataFrame(data, columns = ["Name", "Age"])
print(f'People3:\n {people3}')

print(f'shape: {people3.shape}')
                       
