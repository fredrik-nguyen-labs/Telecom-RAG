from __future__ import annotations

import numpy as np
import pandas as pd

from telecom_rag.kpi import analyze_observation


def test_analyze_observation_produces_dataset_relative_statistics() -> None:
    n = 80
    sinr = np.linspace(-5, 25, n)
    cqi = np.clip(5 + 0.35 * sinr, 1, 15)
    mcs = np.clip(2 + 1.2 * cqi, 0, 28)
    throughput = 8 + 2.5 * sinr + 1.5 * cqi

    reference = pd.DataFrame(
        {
            "observation_id": [f"obs_{i}" for i in range(n)],
            "nr_rsrp_dbm": np.linspace(-112, -75, n),
            "nr_sinr_db": sinr,
            "nr_cqi": cqi,
            "nr_mcs": mcs,
            "nr_ri": np.where(np.arange(n) % 2 == 0, 1, 2),
            "throughput_mbps": throughput,
        }
    )
    row = pd.Series(
        {
            "observation_id": "custom",
            "observation_source": "user-entered",
            "nr_rsrp_dbm": -92.0,
            "nr_sinr_db": 8.0,
            "nr_cqi": 8.0,
            "nr_mcs": 12.0,
            "nr_ri": 2.0,
            "throughput_mbps": 18.0,
        }
    )

    analysis = analyze_observation(row, reference)

    assert analysis["source"] == "user-entered"
    assert analysis["percentiles"]
    assert analysis["throughput_correlations"]
    assert analysis["neighbors"]["k"] >= 3
    assert analysis["multivariate"]["percentile"] is not None
    assert "user-entered values" in analysis["context"]
