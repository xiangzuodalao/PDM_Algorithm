import pytest

from valeo_pdm.transformer.config import DbModelConfigError, _build_db_model_config
from valeo_pdm.transformer.config_resolution import resolve_model_config


def test_null_db_fields_are_omitted_then_defaults_fill_them() -> None:
    db_block = _build_db_model_config(
        None,
        None,
        None,
        "sqlserver",
        '{"seq_len": 4, "label_len": 2, "pred_len": 2}',
    )

    effective, source = resolve_model_config(
        defaults={"model_type": "informer", "freq": "1h", "days_back": 7, "source": "db"},
        yaml_models={
            "EQ": {
                "M": {
                    "model_type": "autoformer",
                    "freq": "15min",
                    "yaml_only": True,
                }
            }
        },
        db_models={"EQ": {"M": db_block}},
        equipment_code="EQ",
        meas_code="M",
    )

    assert source == "db"
    assert effective == {
        "model_type": "informer",
        "freq": "1h",
        "days_back": 7,
        "source": "sqlserver",
        "train_params": {"seq_len": 4, "label_len": 2, "pred_len": 2},
    }
    assert "yaml_only" not in effective


@pytest.mark.parametrize("raw", ["not-json", "[]", "1"])
def test_malformed_db_train_params_are_not_treated_as_missing(raw: str) -> None:
    with pytest.raises(DbModelConfigError):
        _build_db_model_config("informer", "1h", 7, "db", raw)
