# 自定义求解器（SA 等）

## 接口约定

在任意 `.py` 中实现：

```python
def solve(tasks, config, seed) -> list[int]:
    """返回任务下标的排列，如 [2, 0, 1, 3, ...]"""
    ...
```

- `tasks`: `main.Task` 列表  
- `config`: 与 `valid.py` 相同的 `dict` 配置（含 `WEIGHTS`、`DECODE_STRATEGY` 等）  
- `seed`: 可复现随机种子  

**不要**在求解器内单独 decode 或改适应度公式；对比实验由 `benchmark_api.evaluate_permutation` 统一评估。

## 命令行接入

```bash
# 默认 GA + 基线
python valid.py

# 加入 SA（使用模板）
python valid.py --algorithms GA,SA,HERF,WSPT --solver SA=solvers/sa_template.py

# 队友自己的实现
python valid.py --algorithms GA,SA --solver SA=solvers/sa_team.py --workers 6
```

## Python API

```python
from benchmark_api import BenchmarkRunner

runner = BenchmarkRunner(
    primary="GA",
    algorithms=["GA", "SA", "HERF"],
    solvers={"SA": "solvers/sa_team.py"},
    workers=4,
    fast=True,
)
df = runner.run(["datasets/benchmark/medium_seed42.json"], n_runs=5)
```

## 参考

- 模板：`sa_template.py`  
- 核心：`../benchmark_api.py`
