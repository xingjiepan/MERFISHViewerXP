import pytest

from merfishviewerxp.adapters.merlin.positions import load_positions
from merfishviewerxp.errors import DatasetValidationError


def test_headerless_positions(tmp_path):
    path = tmp_path / "positions.csv"
    path.write_text("1.5,2.5\n3.5,4.5\n5.5,6.5\n")
    df = load_positions(path)
    assert list(df.index) == [0, 1, 2]
    assert df.loc[1, "x_um"] == pytest.approx(3.5)
    assert df.loc[1, "y_um"] == pytest.approx(4.5)


def test_headered_positions_with_aliases(tmp_path):
    path = tmp_path / "positions.csv"
    path.write_text("stage_x,stage_y\n1.0,2.0\n3.0,4.0\n")
    df = load_positions(path)
    assert df.loc[0, "x_um"] == pytest.approx(1.0)
    assert df.loc[1, "y_um"] == pytest.approx(4.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetValidationError):
        load_positions(tmp_path / "does_not_exist.csv")


def test_headerless_wrong_column_count_raises(tmp_path):
    path = tmp_path / "positions.csv"
    path.write_text("1.0,2.0,3.0\n")
    with pytest.raises(DatasetValidationError):
        load_positions(path)
