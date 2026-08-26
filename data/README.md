# 数据接口

本仓库不提交患者数据或大体积 UNI2-h 特征。通过环境变量提供：

- `UNI2H_ROOT/<cancer>/uni2-h/pt_files/*.pt`
- `DCT_DATA_CSV_ROOT/clinical/all/<cancer>.csv`
- `DCT_DATA_CSV_ROOT/splits/5fold_uni2h/<cancer>/fold_0.csv ... fold_4.csv`
- `DCT_DATA_CSV_ROOT/signatures/*.csv` 及模型所需组学表

正式实验前生成 `split_manifest.json`，记录每个 split 文件的 SHA-256、患者数量、事件数量、交集检查和一次性外层测试规则。

