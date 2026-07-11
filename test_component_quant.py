from argparse import Namespace

import torch
import torch.nn as nn

from hnerv_utils import dequant_tensor, pack_nbit_tensor, quant_tensor, serialize_quant_tensor, unpack_nbit_tensor
from model_all import HNeRV
from model_chroma_hnerv import ChromaHNeRV420, RGBSplitHNeRV
from train_chroma_hnerv import classify_quant_group, quant_model


def test_pack_roundtrip():
    torch.manual_seed(3)
    for bits in (2, 3, 4, 6, 8):
        for shape in ((1,), (7,), (3, 5), (2, 3, 7)):
            values = torch.randint(0, 2**bits, shape, dtype=torch.uint8)
            packed = pack_nbit_tensor(values, bits)
            restored = unpack_nbit_tensor(packed, bits, values.numel(), values.shape)
            assert torch.equal(values, restored)


def test_quantization_roundtrip():
    torch.manual_seed(4)
    for bits in (2, 3, 4, 6, 8):
        for tensor in (torch.randn(5, 7), torch.full((3, 5), 2.25)):
            record, dequantized = quant_tensor(tensor, bits, quant_axis=0)
            assert torch.isfinite(dequantized).all()
            assert record["quant"].max().item() <= 2**bits - 1
            packed_record = serialize_quant_tensor(record)
            packed_dequantized = dequant_tensor(packed_record)
            assert packed_dequantized.shape == tensor.shape
            assert torch.equal(dequant_tensor(record), packed_dequantized)


def _model_shell(model_type, groups):
    model = model_type.__new__(model_type)
    nn.Module.__init__(model)
    model.encoder = nn.Linear(2, 2)
    for name in groups:
        setattr(model, name, nn.Linear(2, 2))
    return model


def test_group_classification():
    args = Namespace()
    models = [
        (_model_shell(HNeRV, ["decoder", "head_layer"]), {"decoder.": "model", "head_layer.": "model"}),
        (_model_shell(RGBSplitHNeRV, ["shared_decoder", "rgb_branch", "rgb_head"]),
         {"shared_decoder.": "shared", "rgb_branch.": "rgb", "rgb_head.": "rgb"}),
        (_model_shell(ChromaHNeRV420, ["shared_decoder", "y_branch", "y_head", "cbcr_branch", "cbcr_head"]),
         {"shared_decoder.": "shared", "y_branch.": "y", "y_head.": "y",
          "cbcr_branch.": "chroma", "cbcr_head.": "chroma"}),
    ]
    for model, expected_prefixes in models:
        classified = 0
        for name, parameter in model.named_parameters():
            group = classify_quant_group(model, name, args)
            if group == "encoder":
                continue
            expected = [value for prefix, value in expected_prefixes.items() if name.startswith(prefix)]
            assert expected == [group]
            classified += parameter.numel()
        assert classified == sum(parameter.numel() for name, parameter in model.named_parameters()
                                 if not name.startswith("encoder."))


def _quant_args(scheme="uniform", shared=8, y=8, chroma=8, rgb=8):
    return Namespace(
        quant_scheme=scheme, quant_model_bit=8, quant_embed_bit=6, quant_axis=0,
        quant_shared_bit=shared, quant_y_bit=y, quant_chroma_bit=chroma, quant_rgb_bit=rgb,
    )


def test_uniform_regression_and_storage():
    model = _model_shell(ChromaHNeRV420, [
        "shared_decoder", "y_branch", "y_head", "cbcr_branch", "cbcr_head"])
    _, uniform_checkpoint, uniform_report = quant_model(model, _quant_args("uniform"))
    _, component_checkpoint, component_report = quant_model(model, _quant_args("component"))
    assert uniform_checkpoint.keys() == component_checkpoint.keys()
    for name in uniform_checkpoint:
        assert torch.equal(dequant_tensor(uniform_checkpoint[name]), dequant_tensor(component_checkpoint[name]))
    assert uniform_report["storage"]["quantized_param_count"] == component_report["storage"]["quantized_param_count"]

    _, _, mixed_report = quant_model(model, _quant_args("component", shared=8, y=6, chroma=2))
    storage = mixed_report["storage"]
    expected = (
        storage["shared_param_count"] * 8
        + storage["y_param_count"] * 6
        + storage["chroma_param_count"] * 2
    )
    actual = storage["shared_payload_bits"] + storage["y_payload_bits"] + storage["chroma_payload_bits"]
    assert actual == expected
    packed_bits = sum(pack_nbit_tensor(record["quant"], record["bits"]).numel() * 8
                      for record in quant_model(model, _quant_args("component", 8, 6, 2))[1].values())
    assert packed_bits == actual + storage["tensor_padding_bits"]


if __name__ == "__main__":
    test_pack_roundtrip()
    test_quantization_roundtrip()
    test_group_classification()
    test_uniform_regression_and_storage()
    print("component quantization tests passed")
