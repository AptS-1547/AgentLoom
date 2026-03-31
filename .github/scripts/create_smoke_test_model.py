from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def add_expanded_output(
    *,
    prefix: str,
    batch_dim_name: str,
    width: int,
    values: list[float],
    output_name: str,
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    outputs: list[onnx.ValueInfoProto],
) -> None:
    row_name = f"{prefix}_row"
    dim_name = f"{prefix}_dim"
    shape_name = f"{prefix}_shape"

    initializers.append(
        numpy_helper.from_array(
            np.asarray(values, dtype=np.float32).reshape(1, width),
            name=row_name,
        )
    )
    initializers.append(
        numpy_helper.from_array(np.asarray([width], dtype=np.int64), name=dim_name)
    )

    nodes.append(
        helper.make_node(
            "Concat",
            inputs=[batch_dim_name, dim_name],
            outputs=[shape_name],
            axis=0,
            name=f"{prefix}_shape_concat",
        )
    )
    nodes.append(
        helper.make_node(
            "Expand",
            inputs=[row_name, shape_name],
            outputs=[output_name],
            name=f"{prefix}_expand",
        )
    )
    outputs.append(
        helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["batch", width])
    )


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    outputs: list[onnx.ValueInfoProto] = []

    inputs = [
        helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["batch", "seq_len"]
        ),
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch", "seq_len"]
        ),
        helper.make_tensor_value_info(
            "personality", TensorProto.FLOAT, ["batch", 11]
        ),
    ]

    initializers.append(
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="batch_index")
    )

    nodes.append(
        helper.make_node(
            "Shape",
            inputs=["input_ids"],
            outputs=["input_ids_shape"],
            name="input_shape",
        )
    )
    nodes.append(
        helper.make_node(
            "Gather",
            inputs=["input_ids_shape", "batch_index"],
            outputs=["batch_dim"],
            axis=0,
            name="batch_dim",
        )
    )

    add_expanded_output(
        prefix="emotion",
        batch_dim_name="batch_dim",
        width=10,
        values=[0.2, 0.1, -0.1, -0.3, 0.0, -0.2, 0.4, 0.5, -0.4, 0.3],
        output_name="emotion_logits",
        nodes=nodes,
        initializers=initializers,
        outputs=outputs,
    )
    add_expanded_output(
        prefix="behavior",
        batch_dim_name="batch_dim",
        width=12,
        values=[0.3, -0.1, 0.5, 0.0, -0.2, 0.7, -0.4, 0.2, -0.3, 0.1, -0.5, 0.4],
        output_name="behavior_logits",
        nodes=nodes,
        initializers=initializers,
        outputs=outputs,
    )
    add_expanded_output(
        prefix="tone",
        batch_dim_name="batch_dim",
        width=8,
        values=[0.1, 0.4, -0.2, 0.0, 0.3, -0.1, -0.4, 0.2],
        output_name="tone_logits",
        nodes=nodes,
        initializers=initializers,
        outputs=outputs,
    )
    add_expanded_output(
        prefix="intensity",
        batch_dim_name="batch_dim",
        width=1,
        values=[0.65],
        output_name="intensity",
        nodes=nodes,
        initializers=initializers,
        outputs=outputs,
    )
    add_expanded_output(
        prefix="response_length",
        batch_dim_name="batch_dim",
        width=3,
        values=[-0.2, 0.6, 0.1],
        output_name="response_length_logits",
        nodes=nodes,
        initializers=initializers,
        outputs=outputs,
    )

    graph = helper.make_graph(
        nodes=nodes,
        name="bert_inference_smoke_model",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="agent-backend-predict-ci",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a tiny ONNX model for CI smoke tests.")
    parser.add_argument(
        "--output",
        default="build/smoke/bert_smoke_test.onnx",
        help="Output ONNX file path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    onnx.save(model, output_path)
    print(f"Wrote smoke-test model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
