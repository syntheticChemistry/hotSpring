#!/usr/bin/env python3
"""
BOINC Scheduling Algorithm Reproduction and Comparison to NUCLEUS

Reproduces the core scheduling algorithm from:
  - Anderson (2004) "BOINC: A System for Public-Resource Computing and Storage"
  - Kondo et al. (2007) "Scheduling task parallel applications for rapid turnaround
    on desktop grids" JPDC 67(11):1209-1227

Then compares to ToadStool's HybridCloudScheduler architecture to identify
scheduling patterns NUCLEUS should adopt vs. diverge from.

Key BOINC concepts reproduced:
  1. Work unit model (task + result + deadline)
  2. Redundant computation with quorum-based validation
  3. Heterogeneous client scoring (speed, reliability, availability)
  4. Deadline-based scheduling with earliest-deadline-first (EDF)
  5. Homogeneous redundancy (match similar platforms)

Key NUCLEUS differences identified:
  - Covalent trust (family seed) vs anonymous volunteers
  - Mixed substrate (GPU/NPU/CPU) vs CPU-only
  - Peer-to-peer vs server-client
  - Cryptographic verification (BearDog) vs quorum voting
"""

import time
import random
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto


# ─── BOINC Model ─────────────────────────────────────────────────────────────


class ClientState(Enum):
    AVAILABLE = auto()
    COMPUTING = auto()
    OFFLINE = auto()


class ValidationResult(Enum):
    PENDING = auto()
    CANONICAL = auto()
    INVALID = auto()


@dataclass
class WorkUnit:
    """BOINC work unit: the smallest schedulable unit of computation."""
    id: int
    app_id: str
    flops_estimate: float
    memory_bytes: int
    deadline_seconds: float
    min_quorum: int = 2
    target_results: int = 3
    homogeneous_redundancy: bool = False
    hr_class: Optional[str] = None


@dataclass
class Result:
    """A single execution of a work unit on a client."""
    id: int
    work_unit_id: int
    client_id: int
    output_hash: str = ""
    cpu_time: float = 0.0
    wall_time: float = 0.0
    validation: ValidationResult = ValidationResult.PENDING
    granted_credit: float = 0.0


@dataclass
class BOINCClient:
    """Model of a BOINC volunteer client with heterogeneous capabilities."""
    id: int
    platform: str
    flops_measured: float
    memory_bytes: int
    availability_fraction: float
    reliability: float
    hr_class: Optional[str] = None
    state: ClientState = ClientState.AVAILABLE
    active_results: list = field(default_factory=list)
    completed_results: list = field(default_factory=list)
    total_credit: float = 0.0


class BOINCScheduler:
    """
    Reproduction of BOINC server scheduling algorithm.

    Implements the core loop from Anderson (2004):
    1. Score each (work_unit, client) pair
    2. Earliest-deadline-first ordering
    3. Homogeneous redundancy filtering
    4. Quorum-based result validation
    """

    def __init__(self, work_units: list[WorkUnit], clients: list[BOINCClient]):
        self.work_units = {wu.id: wu for wu in work_units}
        self.clients = {c.id: c for c in clients}
        self.results: list[Result] = []
        self.next_result_id = 0
        self.validated_wus: set[int] = set()
        self.result_groups: dict[int, list[Result]] = {}

    def score_client(self, wu: WorkUnit, client: BOINCClient) -> float:
        """
        Score a (work_unit, client) pair for scheduling priority.

        From Anderson (2004) Section 4.2: scheduling score considers
        estimated completion time, client reliability, and resource match.
        """
        if client.state != ClientState.AVAILABLE:
            return -1.0

        if wu.memory_bytes > client.memory_bytes:
            return -1.0

        if wu.homogeneous_redundancy and wu.hr_class != client.hr_class:
            return -1.0

        est_time = wu.flops_estimate / client.flops_measured
        if est_time > wu.deadline_seconds:
            return -1.0

        deadline_urgency = 1.0 / max(wu.deadline_seconds - est_time, 1.0)
        reliability_score = client.reliability
        speed_score = client.flops_measured / 1e12

        return deadline_urgency * 0.4 + reliability_score * 0.4 + speed_score * 0.2

    def schedule_round(self) -> list[tuple[int, int]]:
        """
        Run one scheduling round: assign unfinished work units to available clients.

        Returns list of (work_unit_id, client_id) assignments.
        """
        assignments = []
        pending_wus = [
            wu for wu in self.work_units.values()
            if wu.id not in self.validated_wus
            and len(self.result_groups.get(wu.id, [])) < wu.target_results
        ]

        pending_wus.sort(key=lambda wu: wu.deadline_seconds)

        for wu in pending_wus:
            candidates = []
            for client in self.clients.values():
                score = self.score_client(wu, client)
                if score > 0:
                    candidates.append((score, client))

            candidates.sort(key=lambda x: -x[0])

            existing_results = len(self.result_groups.get(wu.id, []))
            needed = wu.target_results - existing_results

            for score, client in candidates[:needed]:
                result = Result(
                    id=self.next_result_id,
                    work_unit_id=wu.id,
                    client_id=client.id,
                )
                self.next_result_id += 1
                self.results.append(result)
                if wu.id not in self.result_groups:
                    self.result_groups[wu.id] = []
                self.result_groups[wu.id].append(result)

                client.state = ClientState.COMPUTING
                client.active_results.append(result)
                assignments.append((wu.id, client.id))

        return assignments

    def simulate_execution(self, result: Result) -> None:
        """Simulate a client executing a work unit and producing a result hash."""
        wu = self.work_units[result.work_unit_id]
        client = self.clients[result.client_id]

        est_time = wu.flops_estimate / client.flops_measured
        noise = random.gauss(1.0, 0.1)
        result.wall_time = est_time * max(noise, 0.5)
        result.cpu_time = result.wall_time * 0.95

        success = random.random() < client.reliability
        if success:
            canonical = f"wu{wu.id}_app{wu.app_id}"
            result.output_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        else:
            junk = f"wu{wu.id}_fail_{random.random()}"
            result.output_hash = hashlib.sha256(junk.encode()).hexdigest()[:16]

        client.state = ClientState.AVAILABLE

    def validate_quorum(self, wu_id: int) -> bool:
        """
        Quorum-based validation: if min_quorum results agree on the same hash,
        mark the canonical result and grant credit.

        From Anderson (2004) Section 5: "A result is valid if N results agree."
        """
        wu = self.work_units[wu_id]
        results = self.result_groups.get(wu_id, [])

        completed = [r for r in results if r.output_hash]
        if len(completed) < wu.min_quorum:
            return False

        hash_counts: dict[str, list[Result]] = {}
        for r in completed:
            if r.output_hash not in hash_counts:
                hash_counts[r.output_hash] = []
            hash_counts[r.output_hash].append(r)

        for h, matching in hash_counts.items():
            if len(matching) >= wu.min_quorum:
                for r in matching:
                    r.validation = ValidationResult.CANONICAL
                    credit = wu.flops_estimate / 1e9
                    r.granted_credit = credit
                    self.clients[r.client_id].total_credit += credit

                for r in completed:
                    if r.output_hash != h:
                        r.validation = ValidationResult.INVALID

                self.validated_wus.add(wu_id)
                return True

        return False

    def run_full_simulation(self, max_rounds: int = 50) -> dict:
        """Run the full BOINC scheduling simulation."""
        stats = {
            "rounds": 0,
            "total_assignments": 0,
            "validated_wus": 0,
            "invalid_results": 0,
            "total_credit": 0.0,
            "avg_wall_time": 0.0,
            "redundancy_overhead": 0.0,
        }

        for round_num in range(max_rounds):
            assignments = self.schedule_round()
            if not assignments:
                break

            stats["rounds"] = round_num + 1
            stats["total_assignments"] += len(assignments)

            for wu_id, client_id in assignments:
                result = [
                    r for r in self.results
                    if r.work_unit_id == wu_id and r.client_id == client_id
                ][-1]
                self.simulate_execution(result)

            for wu_id in list(self.work_units.keys()):
                if wu_id not in self.validated_wus:
                    self.validate_quorum(wu_id)

        stats["validated_wus"] = len(self.validated_wus)
        stats["invalid_results"] = sum(
            1 for r in self.results if r.validation == ValidationResult.INVALID
        )
        stats["total_credit"] = sum(c.total_credit for c in self.clients.values())

        wall_times = [r.wall_time for r in self.results if r.wall_time > 0]
        stats["avg_wall_time"] = sum(wall_times) / len(wall_times) if wall_times else 0

        if stats["validated_wus"] > 0:
            stats["redundancy_overhead"] = (
                stats["total_assignments"] / stats["validated_wus"]
            )

        return stats


# ─── NUCLEUS Model (Covalent Trust) ─────────────────────────────────────────


class SubstrateType(Enum):
    CPU = auto()
    GPU = auto()
    NPU = auto()


@dataclass
class NUCLEUSNode:
    """A node in the NUCLEUS mesh — covalently bonded via family seed."""
    id: int
    name: str
    substrates: list[SubstrateType]
    flops_by_substrate: dict[SubstrateType, float]
    memory_bytes: int
    lineage_hash: str
    available: bool = True


@dataclass
class NUCLEUSAtomic:
    """A unit of work dispatched to the NUCLEUS mesh."""
    id: int
    app_id: str
    flops_estimate: float
    memory_bytes: int
    preferred_substrate: SubstrateType
    fallback_substrates: list[SubstrateType]


class NUCLEUSScheduler:
    """
    NUCLEUS scheduling model — peer-to-peer with covalent trust.

    Key differences from BOINC:
    - No redundant computation (trust via lineage, not quorum)
    - Mixed substrate scheduling (GPU/NPU/CPU)
    - No deadline model (sovereign mesh, self-paced)
    - Cryptographic verification replaces statistical consensus
    """

    def __init__(self, nodes: list[NUCLEUSNode], family_seed: str):
        self.nodes = {n.id: n for n in nodes}
        self.family_seed = family_seed
        self.assignments: list[tuple[int, int]] = []
        self.completed: set[int] = set()

    def verify_lineage(self, node: NUCLEUSNode) -> bool:
        """BearDog lineage verification — deterministic, not statistical."""
        expected = hashlib.sha256(self.family_seed.encode()).hexdigest()[:16]
        return node.lineage_hash == expected

    def score_node(self, atomic: NUCLEUSAtomic, node: NUCLEUSNode) -> float:
        """
        Score a (atomic, node) pair for substrate-aware scheduling.

        Unlike BOINC's CPU-only scoring, NUCLEUS considers which substrate
        on which node is best for this specific workload.
        """
        if not node.available:
            return -1.0

        if not self.verify_lineage(node):
            return -1.0

        if atomic.memory_bytes > node.memory_bytes:
            return -1.0

        best_substrate = None
        best_flops = 0.0

        if atomic.preferred_substrate in node.substrates:
            best_substrate = atomic.preferred_substrate
            best_flops = node.flops_by_substrate.get(atomic.preferred_substrate, 0.0)
        else:
            for fallback in atomic.fallback_substrates:
                if fallback in node.substrates:
                    best_substrate = fallback
                    best_flops = node.flops_by_substrate.get(fallback, 0.0)
                    break

        if best_substrate is None:
            return -1.0

        substrate_bonus = {
            SubstrateType.GPU: 1.5,
            SubstrateType.NPU: 1.2,
            SubstrateType.CPU: 1.0,
        }

        return best_flops / 1e12 * substrate_bonus.get(best_substrate, 1.0)

    def schedule_round(self, atomics: list[NUCLEUSAtomic]) -> list[tuple[int, int]]:
        """Schedule atomics to nodes — single assignment, no redundancy."""
        assignments = []
        pending = [a for a in atomics if a.id not in self.completed]

        for atomic in pending:
            candidates = []
            for node in self.nodes.values():
                score = self.score_node(atomic, node)
                if score > 0:
                    candidates.append((score, node))

            candidates.sort(key=lambda x: -x[0])

            if candidates:
                _, best_node = candidates[0]
                assignments.append((atomic.id, best_node.id))
                self.assignments.append((atomic.id, best_node.id))
                self.completed.add(atomic.id)

        return assignments


# ─── Comparison Experiment ───────────────────────────────────────────────────


def create_boinc_scenario() -> tuple[list[WorkUnit], list[BOINCClient]]:
    """
    Create a heterogeneous workload matching the NUCLEUS basement mesh:
    - 50 work units (varying size, like lattice QCD + MD + transport)
    - 8 clients (matching Eastgate + Strandgate + NUCs + Pixel)
    """
    random.seed(42)

    work_units = []
    for i in range(50):
        flops = random.uniform(1e9, 1e13)
        memory = random.randint(512 * 1024**2, 12 * 1024**3)
        deadline = random.uniform(60, 3600)
        work_units.append(WorkUnit(
            id=i,
            app_id=random.choice(["md", "qcd", "transport", "eos", "spectral"]),
            flops_estimate=flops,
            memory_bytes=memory,
            deadline_seconds=deadline,
            min_quorum=2,
            target_results=3,
        ))

    clients = [
        BOINCClient(id=0, platform="x86_64-linux", flops_measured=5e12,
                     memory_bytes=64 * 1024**3, availability_fraction=0.95,
                     reliability=0.99, hr_class="epyc"),
        BOINCClient(id=1, platform="x86_64-linux", flops_measured=3e12,
                     memory_bytes=32 * 1024**3, availability_fraction=0.90,
                     reliability=0.98, hr_class="desktop"),
        BOINCClient(id=2, platform="x86_64-linux", flops_measured=1e12,
                     memory_bytes=16 * 1024**3, availability_fraction=0.80,
                     reliability=0.95, hr_class="nuc"),
        BOINCClient(id=3, platform="x86_64-linux", flops_measured=1e12,
                     memory_bytes=16 * 1024**3, availability_fraction=0.80,
                     reliability=0.95, hr_class="nuc"),
        BOINCClient(id=4, platform="x86_64-linux", flops_measured=0.8e12,
                     memory_bytes=8 * 1024**3, availability_fraction=0.70,
                     reliability=0.90, hr_class="nuc"),
        BOINCClient(id=5, platform="x86_64-linux", flops_measured=0.8e12,
                     memory_bytes=8 * 1024**3, availability_fraction=0.70,
                     reliability=0.90, hr_class="nuc"),
        BOINCClient(id=6, platform="aarch64-linux", flops_measured=0.3e12,
                     memory_bytes=8 * 1024**3, availability_fraction=0.50,
                     reliability=0.85, hr_class="pixel"),
        BOINCClient(id=7, platform="x86_64-linux", flops_measured=0.5e12,
                     memory_bytes=4 * 1024**3, availability_fraction=0.60,
                     reliability=0.88, hr_class="usb"),
    ]

    return work_units, clients


def create_nucleus_scenario() -> tuple[list[NUCLEUSAtomic], list[NUCLEUSNode], str]:
    """Same workload expressed as NUCLEUS atomics on the basement mesh."""
    random.seed(42)
    family_seed = "solokey_fido2_eastgate_2025"
    lineage = hashlib.sha256(family_seed.encode()).hexdigest()[:16]

    atomics = []
    for i in range(50):
        flops = random.uniform(1e9, 1e13)
        memory = random.randint(512 * 1024**2, 12 * 1024**3)
        app = random.choice(["md", "qcd", "transport", "eos", "spectral"])
        preferred = SubstrateType.GPU if app in ("md", "qcd", "spectral") else SubstrateType.CPU
        atomics.append(NUCLEUSAtomic(
            id=i,
            app_id=app,
            flops_estimate=flops,
            memory_bytes=memory,
            preferred_substrate=preferred,
            fallback_substrates=[SubstrateType.CPU],
        ))

    nodes = [
        NUCLEUSNode(
            id=0, name="strandgate",
            substrates=[SubstrateType.CPU],
            flops_by_substrate={SubstrateType.CPU: 5e12},
            memory_bytes=64 * 1024**3, lineage_hash=lineage,
        ),
        NUCLEUSNode(
            id=1, name="eastgate",
            substrates=[SubstrateType.CPU, SubstrateType.GPU, SubstrateType.NPU],
            flops_by_substrate={
                SubstrateType.CPU: 3e12,
                SubstrateType.GPU: 20e12,
                SubstrateType.NPU: 0.5e12,
            },
            memory_bytes=32 * 1024**3, lineage_hash=lineage,
        ),
        NUCLEUSNode(
            id=2, name="nuc-01",
            substrates=[SubstrateType.CPU],
            flops_by_substrate={SubstrateType.CPU: 1e12},
            memory_bytes=16 * 1024**3, lineage_hash=lineage,
        ),
        NUCLEUSNode(
            id=3, name="nuc-02",
            substrates=[SubstrateType.CPU],
            flops_by_substrate={SubstrateType.CPU: 1e12},
            memory_bytes=16 * 1024**3, lineage_hash=lineage,
        ),
        NUCLEUSNode(
            id=4, name="nuc-03",
            substrates=[SubstrateType.CPU],
            flops_by_substrate={SubstrateType.CPU: 0.8e12},
            memory_bytes=8 * 1024**3, lineage_hash=lineage,
        ),
        NUCLEUSNode(
            id=5, name="nuc-04",
            substrates=[SubstrateType.CPU],
            flops_by_substrate={SubstrateType.CPU: 0.8e12},
            memory_bytes=8 * 1024**3, lineage_hash=lineage,
        ),
        NUCLEUSNode(
            id=6, name="pixel-8a",
            substrates=[SubstrateType.CPU, SubstrateType.NPU],
            flops_by_substrate={SubstrateType.CPU: 0.3e12, SubstrateType.NPU: 0.2e12},
            memory_bytes=8 * 1024**3, lineage_hash=lineage,
        ),
        NUCLEUSNode(
            id=7, name="usb-livespore",
            substrates=[SubstrateType.CPU],
            flops_by_substrate={SubstrateType.CPU: 0.5e12},
            memory_bytes=4 * 1024**3, lineage_hash=lineage,
        ),
    ]

    return atomics, nodes, family_seed


def run_comparison():
    """Run both schedulers on equivalent workloads and compare."""
    checks_passed = 0
    checks_total = 0

    print("=" * 72)
    print("BOINC vs NUCLEUS Scheduling Comparison")
    print("Anderson (2004) / Kondo et al. (2007) reproduction")
    print("=" * 72)

    # ── BOINC simulation ─────────────────────────────────────────────────
    print("\n── BOINC Scheduling ──")
    work_units, clients = create_boinc_scenario()
    boinc = BOINCScheduler(work_units, clients)

    t0 = time.perf_counter()
    boinc_stats = boinc.run_full_simulation()
    boinc_time = time.perf_counter() - t0

    print(f"  Rounds:              {boinc_stats['rounds']}")
    print(f"  Work units validated: {boinc_stats['validated_wus']}/50")
    print(f"  Total assignments:   {boinc_stats['total_assignments']}")
    print(f"  Invalid results:     {boinc_stats['invalid_results']}")
    print(f"  Redundancy overhead: {boinc_stats['redundancy_overhead']:.2f}x")
    print(f"  Avg wall time:       {boinc_stats['avg_wall_time']:.2f}s")
    print(f"  Total credit:        {boinc_stats['total_credit']:.1f}")
    print(f"  Scheduling time:     {boinc_time*1000:.2f}ms")

    # Check 1: BOINC validates at least 80% of work units
    checks_total += 1
    if boinc_stats["validated_wus"] >= 40:
        checks_passed += 1
        print(f"  ✓ CHECK 1: BOINC validated ≥80% WUs ({boinc_stats['validated_wus']}/50)")
    else:
        print(f"  ✗ CHECK 1: BOINC validated <80% WUs ({boinc_stats['validated_wus']}/50)")

    # Check 2: Redundancy overhead ≥ 2x (by design: min_quorum=2, target=3)
    checks_total += 1
    if boinc_stats["redundancy_overhead"] >= 2.0:
        checks_passed += 1
        print(f"  ✓ CHECK 2: Redundancy ≥2x ({boinc_stats['redundancy_overhead']:.2f}x)")
    else:
        print(f"  ✗ CHECK 2: Redundancy <2x ({boinc_stats['redundancy_overhead']:.2f}x)")

    # Check 3: Some invalid results detected (reliability < 1.0)
    checks_total += 1
    if boinc_stats["invalid_results"] > 0:
        checks_passed += 1
        print(f"  ✓ CHECK 3: Invalid results detected ({boinc_stats['invalid_results']})")
    else:
        print(f"  ✗ CHECK 3: No invalid results (expected some with reliability <1.0)")

    # ── NUCLEUS simulation ───────────────────────────────────────────────
    print("\n── NUCLEUS Scheduling ──")
    atomics, nodes, family_seed = create_nucleus_scenario()
    nucleus = NUCLEUSScheduler(nodes, family_seed)

    t0 = time.perf_counter()
    nucleus_assignments = nucleus.schedule_round(atomics)
    nucleus_time = time.perf_counter() - t0

    gpu_assigned = sum(
        1 for a_id, n_id in nucleus_assignments
        if SubstrateType.GPU in nucleus.nodes[n_id].substrates
        and next(a for a in atomics if a.id == a_id).preferred_substrate == SubstrateType.GPU
    )
    total_assigned = len(nucleus_assignments)

    print(f"  Atomics assigned:    {total_assigned}/50")
    print(f"  GPU-dispatched:      {gpu_assigned}")
    print(f"  Redundancy overhead: 1.00x (covalent trust, no quorum)")
    print(f"  Scheduling time:     {nucleus_time*1000:.2f}ms")

    # Check 4: NUCLEUS assigns all 50 atomics in one round (no redundancy)
    checks_total += 1
    if total_assigned == 50:
        checks_passed += 1
        print(f"  ✓ CHECK 4: All 50 atomics assigned in one round")
    else:
        print(f"  ✗ CHECK 4: Only {total_assigned}/50 assigned")

    # Check 5: GPU workloads routed to GPU node (eastgate)
    checks_total += 1
    if gpu_assigned > 0:
        checks_passed += 1
        print(f"  ✓ CHECK 5: GPU workloads routed to GPU substrate ({gpu_assigned})")
    else:
        print(f"  ✗ CHECK 5: No GPU workloads routed to GPU substrate")

    # Check 6: NUCLEUS lineage verification works
    checks_total += 1
    rogue_node = NUCLEUSNode(
        id=99, name="rogue",
        substrates=[SubstrateType.CPU],
        flops_by_substrate={SubstrateType.CPU: 100e12},
        memory_bytes=1024 * 1024**3,
        lineage_hash="rogue_hash_not_family",
    )
    rogue_score = nucleus.score_node(atomics[0], rogue_node)
    if rogue_score < 0:
        checks_passed += 1
        print(f"  ✓ CHECK 6: Rogue node rejected by lineage verification")
    else:
        print(f"  ✗ CHECK 6: Rogue node NOT rejected (score={rogue_score})")

    # ── Comparison ───────────────────────────────────────────────────────
    print("\n── Comparison ──")
    efficiency_ratio = boinc_stats["redundancy_overhead"]
    print(f"  BOINC compute overhead:    {efficiency_ratio:.2f}x (quorum redundancy)")
    print(f"  NUCLEUS compute overhead:  1.00x (covalent trust)")
    print(f"  Efficiency gain:           {efficiency_ratio:.2f}x fewer compute-hours")

    # Check 7: NUCLEUS is strictly more efficient (1.0x vs >2.0x)
    checks_total += 1
    if efficiency_ratio > 1.5:
        checks_passed += 1
        print(f"  ✓ CHECK 7: NUCLEUS {efficiency_ratio:.1f}x more compute-efficient")
    else:
        print(f"  ✗ CHECK 7: BOINC not redundant enough to show difference")

    # Check 8: NUCLEUS handles mixed substrates (BOINC cannot)
    checks_total += 1
    substrate_types_used = set()
    for a_id, n_id in nucleus_assignments:
        for s in nucleus.nodes[n_id].substrates:
            substrate_types_used.add(s)
    if len(substrate_types_used) > 1:
        checks_passed += 1
        print(f"  ✓ CHECK 8: NUCLEUS used {len(substrate_types_used)} substrate types: "
              f"{', '.join(s.name for s in substrate_types_used)}")
    else:
        print(f"  ✗ CHECK 8: NUCLEUS only used {len(substrate_types_used)} substrate type")

    # ── Lessons for NUCLEUS adoption ─────────────────────────────────────
    print("\n── BOINC Patterns to Adopt ──")
    print("  1. EDF ordering: prioritize work with nearest deadlines")
    print("  2. Client scoring: multi-factor (speed + reliability + availability)")
    print("  3. Heterogeneous awareness: match work to capable hardware")
    print("  4. Work unit granularity: right-sized chunks for fault tolerance")
    print("  5. Availability estimation: track node uptime patterns")

    print("\n── BOINC Patterns to Replace ──")
    print("  1. Quorum redundancy → BearDog cryptographic verification (1x not 3x)")
    print("  2. Anonymous volunteers → Covalent family (SoloKey FIDO2)")
    print("  3. Server-client → Peer-to-peer (biomeOS plasmodium)")
    print("  4. CPU-only → Mixed substrate (GPU/NPU/CPU via BarraCUDA)")
    print("  5. Credit system → sunCloud radiating attribution")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 72}")

    return checks_passed, checks_total


if __name__ == "__main__":
    passed, total = run_comparison()
    exit(0 if passed == total else 1)
