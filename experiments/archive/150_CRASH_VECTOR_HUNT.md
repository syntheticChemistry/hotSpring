# Experiment 150: Crash Vector Hunt

**Objective**: Systematically identify the exact BAR0 operation(s) that lock up the system.

## Context

Despite tightening PCIe completion timeouts (root port → 1-10ms, device → 50-100us)
and routing all exp145 operations through ember's MMIO gateway, the system still
freezes during GPU experiments on the GV100 (Titan V) at 0000:03:00.0.

The crashes happen even when exp145 fails early (socket mismatch) and never touches
BAR0 — implicating **glowplug's health check** (direct BAR0 reads every 5s) or
**ember's own post-swap quiesce** as the crash vector.

## Methodology

1. **Stop glowplug** — remove periodic health probes as a variable
2. **Tighten PCIe timeouts** — root port 1-10ms, device 50-100us
3. **Probe via ember RPCs** — each probe logged with fsync BEFORE execution
4. **Escalate systematically** — safe → risky → dangerous operations
5. **If crash**: read log → last attempted operation = crash vector
6. **If survive all**: re-enable glowplug → test if health check is the vector

## Probe Categories

| Phase | Category        | Registers / Operations                        | Risk   |
|-------|----------------|----------------------------------------------|--------|
| P1    | Identity       | BOOT0 (0x0)                                  | None   |
| P2    | PMC            | PMC_ENABLE (0x200), PMC_DEV_ENABLE (0x20C)   | Low    |
| P3    | PTIMER         | 0x9400, 0x9410                               | Low    |
| P4    | PFIFO          | 0x2004, 0x2100, 0x2504                       | Med    |
| P5    | PBDMA          | 0x40108, 0x40148                             | Med    |
| P6    | PFB            | 0x100000, 0x100800                           | Med    |
| P7    | Falcon SEC2    | 0x87000+CPUCTL, SCTL, PC, EXCI              | Med    |
| P8    | Falcon FECS    | 0x409000+CPUCTL, SCTL                        | Med    |
| P9    | Falcon GPCCS   | 0x41A000+CPUCTL, SCTL                        | Med    |
| P10   | LTC/FBPA       | 0x17E200, 0x9A0000                           | High   |
| P11   | PRAMIN read    | window(0x1700)=0, read 0x700000              | High   |
| P12   | PRAMIN write   | write pattern to 0x700000                     | High   |
| P13   | PMC reset      | PMC_ENABLE toggle bit 5 (SEC2)               | DANGER |
| P14   | PRI ring       | 0x120058 (known poisonous)                    | LETHAL |

## Findings

(populated during execution)

## Status

- [ ] Phase P1-P9: Safe probes
- [ ] Phase P10-P12: VRAM probes
- [ ] Phase P13: PMC reset probe
- [ ] Phase P14: PRI ring probe (expected lethal)
- [ ] Glowplug re-enable test
