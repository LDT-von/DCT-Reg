# 服务器适配器接口

每个方法独立 adapter.py，仅整理官方输入/构造参数；不提供虚构的可运行模型替身。完整官方仓库下载后，由服务器依据实际数据和模型入口实现。

`build(spec, method)` 返回 CPU `model`、5 项 `cases`、`metadata`。

每项 case 包含 fold、case_id、raw_wsi（CPU float32 [1,2048,1536]）、raw_omics_path（该患者共同原始组学文件路径）、kwargs（实际官方完整前向参数）、input_audit。input_audit 必须含 same_wsi_values、same_raw_omics、实际 WSI 维数、组学分组名称/每组维数、截断或填充说明；前两个只能在逐项检查一致后填 true。

共同 WSI 指纹算法：SHA256(str((tuple(t.shape),str(t.dtype))).encode()+t.contiguous().view(torch.uint8).numpy().tobytes())。原始组学指纹为共同患者原始组学文件 binary SHA256。共同病人来自 DCT 主表五折各验证集合字典序首位，冻结输入一次，各方法复用。

metadata 必需：source_commit、native_full_prediction_forward=true、native_tokenization（分组来源）、input_dimension_adaptation（实际补丁说明/原参数，若无改动写 none）、weights_kind。训练权重额外包含 checkpoint_path、checkpoint_sha256、strict_checkpoint_load=true；随机初始化写 random_init，不得产生或引用模型准确率。model 必须在CPU，所有输入先存CPU，一次只向GPU移动当前病例。

模型完整 eval forward 必须包含其真正的融合、生存头及默认执行路径；不只测一个线性层、运输模块或预计算 token 后半段。输入兼容性补丁仅允许可配置/首层输入维数设为1536并记录，不能删除层、减少隐藏维数、改变原生槽数或改写OT求解器；加载现有权重不能擅改维数或用 strict=False。无法兼容时登记缺项，不随意投影/裁剪出点。

组学：各方法有原生六签名/通路/原型分组差异，不强迫所有方法使用329通路；共同的是同一患者原始表达。原型拟合/组织预计算需训练折专用资料；记录训练来源，含在特征到风险路径的模型专属变换必须计入。MMP不能只测已压缩token的融合头，而DCT测完整槽聚合。

DCT保留现有配置、完整原生eval路径，并记录与旧profile是否一致；旧profile仅用于排查，不能和这次同机实测混用。新runner是外部原生模型统一计时入口，不是能自动加载所有外部方法的DCT export CLI。

源码附加字段：metadata.repo_root 是实际官方checkout目录（DCT为本地DCT-Reg）；runner核对实际Git HEAD。metadata.source_files为非空列表，每项path+binary sha256，覆盖实际原生模型/底层组件及输入维数改动文件；本机/服务器工作树脏时保留diff与文件快照，不仅记录HEAD。源码补丁与权重加载方案由服务器审查记录。
