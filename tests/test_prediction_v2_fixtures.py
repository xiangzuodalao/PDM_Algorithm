from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from valeo_pdm.prediction_v2.fixture_generator import generate_isolated_fixtures


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = COMPONENT_ROOT / "configs" / "isolated_fixture_manifest.yaml"
EXPECTED_HASHES = {
    "pilot-cnc-vibration": "5feeb31058fe0521f94758faa22214afe4619e466e22cbb8c6fb7ffcf2369562",
    "pilot-injection-pressure": "6e3faef72b69bbd7e9562e871f405a38286075461d1b0c3a5d170c72d7e0f1a9",
    "pilot-robot-position": "11f162e58f9958ed405a8c67dcbb780900321385d8c2eb45122367b11ed5016e",
    "pilot-tightening-torque": "dae2043bc142ad6bb984e296d2cff21f6a93ccba49f1e42c02ca26b6e413d014",
    "pilot-compressor-pressure": "5e2959d7617adb4fad4da45702bbef75f11832f8f7f9818c7f2e662e83bccf38",
    "pilot-eol-pass-rate": "5fe0b2a40d796cbbd5e42d67cce20d195c2c67a75795cc9f4390848f1c119377",
}


def test_generator_creates_only_the_canonical_six_fixture_objects(tmp_path: Path) -> None:
    """A changed identity or serializer must not silently replace a fixed pilot artifact."""
    output_root = tmp_path / "runtime"

    generated = generate_isolated_fixtures(MANIFEST, output_root)

    assert generated == EXPECTED_HASHES
    assert sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    ) == [
        "manifest.runtime.yaml",
        "objects/pilot-cnc-vibration.json",
        "objects/pilot-compressor-pressure.json",
        "objects/pilot-eol-pass-rate.json",
        "objects/pilot-injection-pressure.json",
        "objects/pilot-robot-position.json",
        "objects/pilot-tightening-torque.json",
    ]
    assert not list(output_root.rglob("*.pt"))
    assert not (output_root / "artifacts").exists()
    assert not list(output_root.rglob("*training*"))
    for profile_id, expected_hash in EXPECTED_HASHES.items():
        artifact = (output_root / "objects" / f"{profile_id}.json").read_bytes()
        assert b"\n" not in artifact
        assert hashlib.sha256(artifact).hexdigest() == expected_hash
    runtime = yaml.safe_load((output_root / "manifest.runtime.yaml").read_text(encoding="utf-8"))
    assert runtime["fixture_mode"] == "isolated-pilot"
    assert {entry["model_profile_id"] for entry in runtime["entries"]} == set(EXPECTED_HASHES)


@pytest.mark.parametrize("initial", ["non-empty", "symlink"])
def test_generator_refuses_an_occupied_or_symlinked_output_directory(
    tmp_path: Path, initial: str
) -> None:
    """Overwriting an existing runtime could replace a verified catalog after review."""
    output_root = tmp_path / "runtime"
    if initial == "non-empty":
        output_root.mkdir()
        (output_root / "existing").write_text("x", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        output_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="empty|symlink"):
        generate_isolated_fixtures(MANIFEST, output_root)


def test_cli_prepare_isolated_fixtures_generates_the_runtime(tmp_path: Path) -> None:
    """Removing the CLI command would leave isolated deployments without a safe generator."""
    from valeo_pdm.cli import main

    output_root = tmp_path / "runtime"
    assert (
        main(
            ["prepare-isolated-fixtures", "--manifest", str(MANIFEST), "--output", str(output_root)]
        )
        == 0
    )
    assert (output_root / "manifest.runtime.yaml").is_file()
