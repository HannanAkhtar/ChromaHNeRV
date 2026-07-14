from argparse import Namespace
import os
import tempfile

import torch
import torch.nn as nn

from hnerv_utils import dequant_tensor, pack_nbit_tensor, quant_tensor, serialize_quant_tensor, unpack_nbit_tensor
from model_all import HNeRV
from efficient_nvloader import load_quantized_video_checkpoint
from model_chroma_hnerv import ChromaHNeRV420, RGBSplitHNeRV
from train_chroma_hnerv import classify_quant_group, quant_model


def test_pack_roundtrip():
    torch.manual_seed(3)
    for bits in range(2, 9):
        for shape in ((1,), (7,), (9,), (3, 5), (2, 3, 7)):
            values = torch.randint(0, 2**bits, shape, dtype=torch.uint8)
            packed = pack_nbit_tensor(values, bits)
            restored = unpack_nbit_tensor(packed, bits, values.numel(), values.shape)
            assert torch.equal(values, restored)


def test_quantization_roundtrip():
    torch.manual_seed(4)
    for bits in range(2, 9):
        for axis in (-1, 0):
            for tensor in (torch.randn(5, 7) * 3 - 1, torch.full((3, 5), -2.25)):
                record, dequantized = quant_tensor(tensor, bits, quant_axis=axis)
                assert torch.isfinite(dequantized).all()
                assert record["quant"].min().item() >= 0
                assert record["quant"].max().item() <= 2**bits - 1
                assert record["numel"] * bits == tensor.numel() * bits
                packed_record = serialize_quant_tensor(record)
                assert packed_record["packed"].numel() == (tensor.numel() * bits + 7) // 8
                packed_dequantized = dequant_tensor(packed_record)
                assert packed_dequantized.shape == tensor.shape
                assert torch.equal(dequant_tensor(record), packed_dequantized)


def test_packed_checkpoint_loading_all_bits():
    with tempfile.TemporaryDirectory() as directory:
        for bits in range(2, 9):
            embed, _ = quant_tensor(torch.randn(5, 3), bits, -1)
            weight, _ = quant_tensor(torch.randn(7, 5), bits, 0)
            checkpoint = {
                "format_version": 2,
                "quant_config": {"model_bit": bits, "embed_bit": bits},
                "embed": serialize_quant_tensor(embed),
                "model": {"decoder.weight": serialize_quant_tensor(weight)},
                "buffers": {},
                "storage_report": {},
            }
            path = os.path.join(directory, f"quant_{bits}.pth")
            torch.save(checkpoint, path)
            loaded_embed, loaded_model, _ = load_quantized_video_checkpoint(path)
            assert torch.equal(loaded_embed, dequant_tensor(embed))
            assert torch.equal(loaded_model["decoder.weight"], dequant_tensor(weight))

        legacy_embed, _ = quant_tensor(torch.randn(5, 3), 8, -1)
        legacy_weight, _ = quant_tensor(torch.randn(7, 5), 8, 0)
        legacy_path = os.path.join(directory, "legacy_quant_vid.pth")
        torch.save({"embed": legacy_embed, "model": {"decoder.weight": legacy_weight}}, legacy_path)
        loaded_embed, loaded_model, loaded = load_quantized_video_checkpoint(legacy_path)
        assert loaded.get("format_version", 1) == 1
        assert torch.equal(loaded_embed, dequant_tensor(legacy_embed))
        assert torch.equal(loaded_model["decoder.weight"], dequant_tensor(legacy_weight))


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
    test_packed_checkpoint_loading_all_bits()
    test_group_classification()
    test_uniform_regression_and_storage()
    print("component quantization tests passed")
