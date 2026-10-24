"""准备状态契约测试：必要文件按启用的平台判断。

背景（L-a 现场实测，见 `掩盖台账.md` N-17）：
`sim_579b6e123ba4`（`enable_reddit=false`）的 prepare 日志已写"模拟准备完成"、
`state.json` 为 `ready`/`config_generated=true`，但 `/api/simulation/prepare/status`
一直返回 `not_started` / `already_prepared=false`——
因为检查函数固定要求 `reddit_profiles.json`，而单平台（Twitter）模拟根本不会写它。
"""

import json

from app.api import simulation as simulation_api


def _write_sim(tmp_path, *, twitter=True, reddit=True, status="ready"):
    sim_dir = tmp_path / "sim_test"
    sim_dir.mkdir()
    (sim_dir / "state.json").write_text(
        json.dumps(
            {
                "simulation_id": "sim_test",
                "status": status,
                "config_generated": True,
                "entities_count": 2,
                "entity_types": ["Person"],
                "profiles_count": 2,
                "enable_twitter": twitter,
                "enable_reddit": reddit,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (sim_dir / "simulation_config.json").write_text("{}", encoding="utf-8")
    if twitter:
        (sim_dir / "twitter_profiles.csv").write_text(
            "user_id,name\n0,甲\n1,乙\n", encoding="utf-8"
        )
    if reddit:
        (sim_dir / "reddit_profiles.json").write_text(
            json.dumps([{"name": "甲"}, {"name": "乙"}], ensure_ascii=False),
            encoding="utf-8",
        )
    return sim_dir


def _check(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.config.Config.OASIS_SIMULATION_DATA_DIR", str(tmp_path), raising=False
    )
    return simulation_api._check_simulation_prepared("sim_test")


def test_twitter_only_simulation_is_prepared(monkeypatch, tmp_path):
    _write_sim(tmp_path, twitter=True, reddit=False)
    is_prepared, info = _check(monkeypatch, tmp_path)
    assert is_prepared is True, f"单平台模拟必须被判为已准备，实际: {info}"
    assert info["profiles_count"] == 2, "档案数应从 CSV 统计，而不是固定读 reddit JSON"


def test_reddit_only_simulation_is_prepared(monkeypatch, tmp_path):
    _write_sim(tmp_path, twitter=False, reddit=True)
    is_prepared, info = _check(monkeypatch, tmp_path)
    assert is_prepared is True, f"只跑 Reddit 也应判为已准备，实际: {info}"
    assert info["profiles_count"] == 2


def test_missing_enabled_platform_file_is_not_prepared(monkeypatch, tmp_path):
    """启用了 Reddit 却没写档案文件 → 仍未准备（该报的必须报）。"""

    sim_dir = _write_sim(tmp_path, twitter=True, reddit=True)
    (sim_dir / "reddit_profiles.json").unlink()
    is_prepared, info = _check(monkeypatch, tmp_path)
    assert is_prepared is False
    assert "reddit_profiles.json" in info.get("missing_files", [])


def test_both_platforms_simulation_is_prepared(monkeypatch, tmp_path):
    _write_sim(tmp_path, twitter=True, reddit=True)
    is_prepared, _ = _check(monkeypatch, tmp_path)
    assert is_prepared is True
