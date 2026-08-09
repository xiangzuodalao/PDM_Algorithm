from valeo_pdm.training.registry import get_trainer


def test_mvp_trainer_targets_are_actually_importable_without_running_training() -> None:
    informer = get_trainer("informer")
    autoformer = get_trainer("autoformer")

    assert callable(informer)
    assert callable(autoformer)
    assert informer.__module__ == "valeo_pdm.training.entrypoints"
    assert autoformer.__module__ == "valeo_pdm.training.entrypoints"
