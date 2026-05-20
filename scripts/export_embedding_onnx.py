"""
导出 Sentence-Transformers Embedding 模型到 ONNX

从 HuggingFace 缓存加载已下载的模型，导出为 ONNX 格式供 C++ 推理使用。
支持 sentence-transformers 模型（如 paraphrase-multilingual-MiniLM-L12-v2）。

输出:
- model.onnx: ONNX 模型文件（last_hidden_state 输出，需要外部 pooling）
- model_metadata.json: 元数据（维度、pooling 策略等）
- tokenizer.json: 已存在于 HF 缓存中，无需重新导出

用法:
    python export_embedding_onnx.py \
        --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --output ./onnx_embedding \
        --opset 14
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def export_sentence_transformer_onnx(
    model_name: str,
    output_dir: str,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    simplify: bool = True,
    max_length: int = 128
) -> Tuple[str, str]:
    """
    导出 sentence-transformers 模型到 ONNX

    Args:
        model_name: HuggingFace 模型名（如 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2）
        output_dir: 输出目录
        opset_version: ONNX opset 版本
        dynamic_batch: 是否支持动态 batch
        simplify: 是否简化 ONNX 图
        max_length: 最大序列长度

    Returns:
        (onnx_path, metadata_path)
    """
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    print(f"加载模型: {model_name}")

    # 加载模型和配置
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Max position embeddings: {config.max_position_embeddings}")

    # 准备输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出 ONNX
    onnx_path = output_dir / "model.onnx"

    # 创建示例输入
    batch_size = 1
    seq_length = max_length
    dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    # 动态轴配置
    if dynamic_batch:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
        }
    else:
        dynamic_axes = None

    # 包装模型以只输出 last_hidden_state
    class EmbeddingModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # 返回 last_hidden_state [batch, seq, hidden]
            return outputs.last_hidden_state

    wrapped_model = EmbeddingModelWrapper(model)
    wrapped_model.eval()

    print(f"导出 ONNX: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True
        )

    print(f"ONNX 导出成功")

    # 简化 ONNX (可选)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("简化 ONNX 模型...")
            onnx_model = onnx.load(str(onnx_path))
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified_model, str(onnx_path))
                print("简化成功")
            else:
                print("简化验证失败，保留原模型")
        except ImportError:
            print("未安装 onnxsim，跳过简化 (pip install onnxsim)")
        except Exception as e:
            print(f"简化失败: {e}，保留原模型")

    # 导出元数据
    metadata_path = output_dir / "model_metadata.json"

    # 检测 pooling 策略（从 sentence-transformers 配置）
    pooling_mode = "mean"  # 默认
    try:
        pooling_config_path = Path(model_name.replace("/", "--")) / "1_Pooling" / "config.json"
        # 尝试从缓存读取
        from transformers.utils import cached_file
        pooling_file = cached_file(model_name, "1_Pooling/config.json", _raise_exceptions_for_missing_entries=False)
        if pooling_file:
            with open(pooling_file, "r") as f:
                pooling_cfg = json.load(f)
                if pooling_cfg.get("pooling_mode_cls_token"):
                    pooling_mode = "cls"
                elif pooling_cfg.get("pooling_mode_mean_tokens"):
                    pooling_mode = "mean"
    except Exception:
        pass

    metadata = {
        "model_name": model_name,
        "model_type": "sentence_transformer_embedding",
        "hidden_size": config.hidden_size,
        "max_position_embeddings": config.max_position_embeddings,
        "vocab_size": config.vocab_size,
        "pooling_strategy": pooling_mode,
        "normalize": True,
        "output_shape": {
            "last_hidden_state": ["batch_size", "sequence_length", config.hidden_size]
        },
        "usage": {
            "description": "输出 last_hidden_state，需要外部 pooling（mean/cls）+ L2 normalize",
            "pooling_note": f"推荐使用 {pooling_mode} pooling",
            "normalize_note": "sentence-transformers 模型通常需要 L2 normalize"
        }
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"元数据保存到: {metadata_path}")

    # 检查 tokenizer.json 是否存在
    try:
        from transformers.utils import cached_file
        tokenizer_file = cached_file(model_name, "tokenizer.json")
        if tokenizer_file:
            print(f"tokenizer.json 已存在于 HF 缓存: {tokenizer_file}")
            print(f"  可直接使用该文件，无需重新导出")
    except Exception:
        print("未找到 tokenizer.json，可能需要手动导出")

    return str(onnx_path), str(metadata_path)


def verify_onnx(onnx_path: str, model_name: str):
    """
    验证 ONNX 模型输出与 PyTorch 一致

    Args:
        onnx_path: ONNX 模型路径
        model_name: HuggingFace 模型名
    """
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoModel

    print(f"\n验证 ONNX 模型: {onnx_path}")

    # 加载 ONNX
    ort_session = ort.InferenceSession(onnx_path)

    # 加载 PyTorch 模型
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 创建测试输入
    batch_size = 2
    seq_length = 64
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    # PyTorch 推理
    with torch.no_grad():
        pt_output = model(input_ids=input_ids, attention_mask=attention_mask)
        pt_hidden = pt_output.last_hidden_state.numpy()

    # ONNX 推理
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_hidden = ort_outputs[0]

    # 比较
    max_diff = np.abs(pt_hidden - ort_hidden).max()
    mean_diff = np.abs(pt_hidden - ort_hidden).mean()

    print(f"last_hidden_state 最大差异: {max_diff:.6f}")
    print(f"last_hidden_state 平均差异: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("验证通过!")
    else:
        print("存在较大差异，请检查")


def benchmark_onnx(onnx_path: str, batch_sizes: list = [1, 4, 8, 16], num_runs: int = 100):
    """
    ONNX 模型性能测试

    Args:
        onnx_path: ONNX 模型路径
        batch_sizes: 测试的 batch 大小列表
        num_runs: 每个 batch 大小的运行次数
    """
    import numpy as np
    import onnxruntime as ort
    import time

    print(f"\n性能测试: {onnx_path}")

    # 尝试使用 GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        provider = ort_session.get_providers()[0]
        print(f"使用: {provider}")
    except Exception:
        ort_session = ort.InferenceSession(onnx_path)
        print("使用: CPU")

    seq_length = 64

    print(f"\n{'Batch':<10} {'延迟(ms)':<15} {'吞吐量(samples/s)':<20}")
    print("-" * 50)

    for batch_size in batch_sizes:
        # 准备输入
        input_ids = np.random.randint(0, 30000, (batch_size, seq_length), dtype=np.int64)
        attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # 预热
        for _ in range(10):
            ort_session.run(None, inputs)

        # 计时
        start = time.perf_counter()
        for _ in range(num_runs):
            ort_session.run(None, inputs)
        elapsed = time.perf_counter() - start

        avg_latency = (elapsed / num_runs) * 1000  # ms
        throughput = (batch_size * num_runs) / elapsed

        print(f"{batch_size:<10} {avg_latency:<15.2f} {throughput:<20.1f}")


def main():
    parser = argparse.ArgumentParser(description="导出 Sentence-Transformers Embedding 模型到 ONNX")
    parser.add_argument("--model", type=str,
                       default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       help="HuggingFace 模型名")
    parser.add_argument("--output", type=str, default="./onnx_embedding",
                       help="输出目录")
    parser.add_argument("--opset", type=int, default=14,
                       help="ONNX opset 版本")
    parser.add_argument("--max-length", type=int, default=128,
                       help="最大序列长度")
    parser.add_argument("--no-simplify", action="store_true",
                       help="不简化 ONNX")
    parser.add_argument("--verify", action="store_true",
                       help="验证导出结果")
    parser.add_argument("--benchmark", action="store_true",
                       help="性能测试")

    args = parser.parse_args()

    onnx_path, metadata_path = export_sentence_transformer_onnx(
        model_name=args.model,
        output_dir=args.output,
        opset_version=args.opset,
        simplify=not args.no_simplify,
        max_length=args.max_length
    )

    if args.verify:
        verify_onnx(onnx_path, args.model)

    if args.benchmark:
        benchmark_onnx(onnx_path)

    print(f"\n导出完成!")
    print(f"  ONNX 模型: {onnx_path}")
    print(f"  元数据: {metadata_path}")
    print(f"\n使用方法:")
    print(f"  1. 设置环境变量:")
    print(f"     $env:HF_TEXT_EMBEDDING_ONNX = \"{onnx_path}\"")
    print(f"  2. 运行 C++ 测试:")
    print(f"     ctest --test-dir build -C Release -R Embedding")


if __name__ == "__main__":
    main()
