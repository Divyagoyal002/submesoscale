"""The browser port (site/assets/physics.js) must match the Python physics."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from submeso.physics import coriolis, geostrophic_velocity, relative_vorticity

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_physics_matches_python(tmp_path):
    rng = np.random.default_rng(0)
    lat = np.linspace(36.0, 41.0, 60)
    lon = np.linspace(-2.0, 6.0, 90)
    adt = np.cumsum(np.cumsum(rng.normal(size=(60, 90)), 0), 1) * 1e-3
    adt[20:26, 30:40] = np.nan  # an island
    u, v = geostrophic_velocity(adt, lat, lon)
    ro = relative_vorticity(u, v, lat, lon) / coriolis(lat)[:, None]

    def as_list(a):
        return [None if not np.isfinite(x) else float(x) for x in np.ravel(a)]

    ref = tmp_path / "ref.json"
    ref.write_text(
        json.dumps(
            {
                "lat": lat.tolist(),
                "lon": lon.tolist(),
                "adt": as_list(adt),
                "u": as_list(u),
                "v": as_list(v),
                "ro": as_list(ro),
            }
        )
    )
    out = subprocess.run(
        ["node", str(ROOT / "tests/web/check_physics.mjs"), str(ref)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
