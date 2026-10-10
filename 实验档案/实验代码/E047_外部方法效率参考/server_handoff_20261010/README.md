# E047 BLCA 性能—显存/时延 服务器交接包

用户要的是两张性能—推理成本散点图。上一版BLCA均值图和差值图重复表2，用户已明确不接受；本包准备替代它们所需的真实成本测量。当前未生成外部成本图、未运行模型、未改论文Word。

直接把 SERVER_PROMPT.txt 发给服务器助手，并上传整个zip。先核对现有checkpoint，缺权重不训练；如做公开分数＋新成本参照图，可测完整随机初始化官方模型的前向成本，但明确它不是同条件性能复现。真实同条件图仍走现有严格预测审核器。

目标：DCT＋8个外部多模态方法，BLCA DSS，两个散点panel（显存MiB / 耗时sec），星形标DCT。不要预设DCT处于左上角。只下载源码无法获得真实横坐标，至少要执行无梯度前向。

文件：
- SERVER_PROMPT.txt：完整服务器任务指令与已有数据入口。
- benchmark_spec.example.json：9方法、官方固定commit、冻结BLCA均值和空白病例指纹。服务器补数据后另存real.json；空模板不能运行。
- blca_scores_frozen.csv：9方法纵轴来源，包括编码器/划分，防止混E046。
- fetch_official_repos.py：下载8个完整固定commit官方仓库，不执行模型。
- benchmark_native.py：统一GPU前向成本记录器。各外部方法实际适配器需服务器按真实环境编写，不能直接声称本脚本能加载九个模型。
- adapter_contract.md：原生输入/完整前向/指纹/权重与源码边界。
- plot_measured_tradeoff.py：只接受27个实测profile，生成用户要的双panel和CSV/audit；不从论文图片抄横坐标。
- official_source_files/：18个已有官方文件及来源清单，仅参考，完整依赖由下载脚本获取。

本机核验限于Python语法、方法/分数/commit快照、缺数据拒绝和计时/绘图数据接口；未执行GPU测量，未评估准确率。服务器成功测量并回传图前，本任务为待测。

run_benchmarks.py 默认仅打印测量计划；--execute 才启动27个独立前向进程。test_handoff_contract.py只验证JSON数据检查器，不导入PyTorch、不执行模型。
