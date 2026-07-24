from uvg_utils import decoder_stage_resolutions, split_resolution_metadata


def test_uvg_decoder_and_split_resolutions():
    strides = [5, 4, 4, 3, 2]
    assert decoder_stage_resolutions(960, 1920, strides) == [
        (10, 20), (40, 80), (160, 320), (480, 960), (960, 1920)]
    a320 = split_resolution_metadata("960_1920", strides, "a320")
    assert (a320["shared_output_height"], a320["shared_output_width"]) == (480, 960)
    assert (a320["chroma_output_height"], a320["chroma_output_width"]) == (480, 960)
    a160 = split_resolution_metadata("960_1920", strides, "a160")
    assert (a160["shared_output_height"], a160["shared_output_width"]) == (160, 320)
    assert (a160["chroma_output_height"], a160["chroma_output_width"]) == (480, 960)
    assert (a160["final_output_height"], a160["final_output_width"]) == (960, 1920)
