from __future__ import annotations

from pathlib import Path

import soundfile as sf

from music_locator.models import AssetKind
from music_locator.storage import Workspace


def test_assets_are_content_addressed_and_immutable(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    first = workspace.store_asset(
        b"immutable evidence",
        kind=AssetKind.ARTIFACT,
        media_type="text/plain",
        original_name="a.txt",
    )
    second = workspace.store_asset(
        b"immutable evidence",
        kind=AssetKind.ARTIFACT,
        media_type="text/plain",
        original_name="a.txt",
    )
    assert first == second
    blob_path = workspace.asset_path(first.asset_id)
    assert blob_path.read_bytes() == b"immutable evidence"
    assert blob_path.stat().st_mode & 0o222 == 0


def test_existing_wav_can_be_described(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    sf.write(path, [0.0] * 2_205, 22_050, subtype="PCM_16")
    assert sf.info(path).frames == 2_205
