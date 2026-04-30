// SPDX-License-Identifier: AGPL-3.0-or-later

use super::types::BaselineProvenance;

/// Python L1 (SEMF) best χ²/datum — surrogate/nuclear-eos/wrapper/objective.py
pub const L1_PYTHON_CHI2: BaselineProvenance = BaselineProvenance {
    label: "L1 Python best chi2/datum (52 nuclei)",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L1 --nuclei=selected --samples=1000 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 6.62,
    unit: "chi2/datum",
};

/// Python L1 best candidate count
pub const L1_PYTHON_CANDIDATES: BaselineProvenance = BaselineProvenance {
    label: "L1 Python candidate count",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L1 --nuclei=selected --samples=1000 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 1008.0,
    unit: "candidates evaluated",
};

/// Python L2 (HFB) best `χ²`/datum — surrogate/nuclear-eos/wrapper/objective.py
pub const L2_PYTHON_CHI2: BaselineProvenance = BaselineProvenance {
    label: "L2 Python best chi2/datum (52 nuclei)",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L2 --nuclei=selected --samples=100 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 61.87,
    unit: "chi2/datum",
};

/// Python L2 candidate count
pub const L2_PYTHON_CANDIDATES: BaselineProvenance = BaselineProvenance {
    label: "L2 Python candidate count",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L2 --nuclei=selected --samples=100 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 96.0,
    unit: "candidates evaluated",
};

/// Python L2 total `χ²` (un-normalized)
pub const L2_PYTHON_TOTAL_CHI2: BaselineProvenance = BaselineProvenance {
    label: "L2 Python total chi2 (52 nuclei, unnormalized)",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L2 --nuclei=selected --samples=100 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 28_450.0,
    unit: "chi2_total",
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(clippy::assertions_on_constants, reason = "constants sanity check")]
    fn provenance_records_have_content() {
        assert!(!L1_PYTHON_CHI2.script.is_empty());
        assert!(!L1_PYTHON_CHI2.commit.is_empty());
        assert!(!L1_PYTHON_CHI2.command.is_empty());
        assert!(L1_PYTHON_CHI2.value > 0.0);
    }
}
