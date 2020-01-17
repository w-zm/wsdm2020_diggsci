# digsci_wsdm_2020 召回部分

## Install Requirements

```
pip3 install -r requirements.txt
python3 -m spacy download en
```
## Recall
运行以下命令
```
python3 bm25.py -n 200 -data_dir path_to_data
python3 tfidf.py -n 200 -data_dir path_to_data
```
得到针对训练集的召回文件用于下一步构造训练集，以及测试集的召回文件用于下一阶段精排