import lancedb
import os

uri = "./data/sample-lancedb"
table_name = "my_table"
db = lancedb.connect(uri)
# check if the database exists and only proceed if it does not
if db.table_names() == []:
    print("Creating table")
    table = db.create_table(table_name,
                         data=[{"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
                               {"vector": [5.9, 26.5], "item": "bar", "price": 20.0}])
else:
    print("Opening table")
    table = db.open_table(table_name)

result = table.search([100, 100]).limit(2).to_df()
print(result)
