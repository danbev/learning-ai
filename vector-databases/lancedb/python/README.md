## LanceDB example in Python
This is an example of using LanceDB in Python more or less taken from the
LanceDB [Guides](https://lancedb.github.io/lancedb/guides/tables/)

### Running the example
First create a virtual python environment and install the requirements:
```console
$ python -m venv lancev
$ source lancev/bin/activate
(lancev) $ pip install -r requirements.txt
```
Now we should be able to run the example using the following command:
```console
(lancev) $ python src/lancedb-example.py 
Opening table
Table: my_table
        vector item  price     _distance
0  [5.9, 26.5]  bar   20.0  14257.059570
1   [3.1, 4.1]  foo   10.0  18586.421875
```

