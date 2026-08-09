from valeo_pdm.transformer.config_resolution import merge_effective_models, resolve_model_config


def test_db_block_replaces_yaml_whole_then_defaults_fill_top_level() -> None:
    defaults = {"model_type": "informer", "freq": "1h", "source": "db"}
    yaml_models = {
        "EQ": {
            "MEAS": {
                "freq": "15min",
                "yaml_only": "must-not-leak",
                "train_params": {"seq_len": 99, "label_len": 2, "pred_len": 1},
            }
        }
    }
    db_models = {
        "EQ": {
            "MEAS": {
                "source": "sqlserver",
                "db_only": True,
                "train_params": {"seq_len": 4, "label_len": 2, "pred_len": 2},
            }
        }
    }

    effective, source = resolve_model_config(
        defaults=defaults,
        yaml_models=yaml_models,
        db_models=db_models,
        equipment_code="EQ",
        meas_code="MEAS",
    )

    assert source == "db"
    assert effective == {
        "model_type": "informer",
        "freq": "1h",
        "source": "sqlserver",
        "db_only": True,
        "train_params": {"seq_len": 4, "label_len": 2, "pred_len": 2},
    }
    assert "yaml_only" not in effective


def test_merge_effective_models_uses_same_precedence() -> None:
    merged = merge_effective_models(
        defaults={"freq": "1h", "source": "db"},
        yaml_models={"EQ": {"M": {"source": "csv", "yaml_only": True}}},
        db_models={"EQ": {"M": {"db_only": True}}},
    )

    assert merged == {"EQ": {"M": {"freq": "1h", "source": "db", "db_only": True}}}


def test_only_authorized_top_level_defaults_are_inherited() -> None:
    effective, source = resolve_model_config(
        defaults={
            "model_type": "informer",
            "freq": "1h",
            "days_back": 7,
            "source": "db",
            "train_params": {"seq_len": 999},
            "secret": "must-not-leak",
        },
        yaml_models={"EQ": {"M": {"train_params": {"seq_len": 4}}}},
        db_models={},
        equipment_code="EQ",
        meas_code="M",
    )

    assert source == "yaml"
    assert effective == {
        "model_type": "informer",
        "freq": "1h",
        "days_back": 7,
        "source": "db",
        "train_params": {"seq_len": 4},
    }
