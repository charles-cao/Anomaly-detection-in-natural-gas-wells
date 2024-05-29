# 如何使用？

```
pip install -r requirements.txt
```

首先进行配置文件，demo见`./demo_config.json`。

- `name`为算法的名称
- `input path`为输入数据集根路径
- `input file`为输入数据集名称

格式为

```
+-- dataset
|   +-- csv
|   |   +-- gas1.csv
|   |   +-- gas2.csv
|   +-- mat
|   |   +-- gas1.mat
|   |   +-- gas2.mat
```

`.csv`文件是使用逗号分隔的`.csv`文件，倒数第2列为label，倒数第一列为drift标识符，这里置为0。

`.mat`文件是MATLAB的`.mat`文件，`Y`为数据，`L`为label，`C`为drift标识符，这里置为0。

- `output path`为输出结果路径。输出文件的文件名是由下划线分隔的，第一项为算法名称，第二项为AUC指标，第三项为时间戳，输出文件保存了每个点的异常分数。
- `argument`为该算法的参数，详情见`./config`中对应算法的参数。

之后运行
```
python run_algorithm.py -r demo_config.json
```



# How does it work?

```
pip install -r requirements.txt
```

Start with the configuration file, for demo see `. /demo_config.json`.

- `name` is the name of the algorithm
- `input path` is the root path of the input dataset.
- `input file` is the name of the input dataset

The format is

```
+-- dataset
|   +-- csv
|   |   +-- gas1.csv
|   |   +-- gas2.csv
|   +-- mat
|   |   +-- gas1.mat
|   |   +-- gas2.mat
```

The `.csv` file is a comma-delimited `.csv` file, with the penultimate column as a label and the penultimate column as a drift identifier, which is set to 0 here.

The `.mat` file is a MATLAB `.mat` file, `Y` is the data, `L` is the label, and `C` is the drift identifier, which is set to 0 here.

- `output path` is the output result path. The filenames of the output file are separated by underscores, the first item is the name of the algorithm, the second item is the AUC metrics, and the third item is the timestamp, the output file saves the anomaly scores for each point.
- The `argument` is the parameter for the algorithm, as detailed in `. /config` for the parameters of the corresponding algorithm.

Afterwards run
```
python run_algorithm.py -r demo_config.json
```
