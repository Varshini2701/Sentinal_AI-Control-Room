# SUMO scenario — high-fidelity closed-loop demo

Sentinel's control loop runs against a [`TrafficEnvironment`](../libs/sentinel/simulation/environment.py)
port. The default implementation is the dependency-free **analytical twin**
([`AnalyticalTrafficEnvironment`](../libs/sentinel/simulation/analytical.py)), which powers the
tests and the CI benchmark. This directory holds the **SUMO** implementation for the visual,
microsimulated demo — same control code, higher fidelity.

## Layout

```
sim/
├─ network/
│  ├─ intersection.nod.xml   # 5 nodes: signalised centre + 4 approaches
│  ├─ intersection.edg.xml   # 8 edges: in/out per approach
│  └─ routes.rou.xml         # asymmetric demand (heavy N-S, light E-W)
├─ build_network.py          # netconvert + generates the TLS program our adapter expects
└─ output/                   # generated artifacts (git-ignored)
```

## Prerequisites

1. Install [SUMO](https://sumo.dlr.de/docs/Installing/index.html) (≥ 1.15).
2. Set `SUMO_HOME` (e.g. `C:\Program Files (x86)\Eclipse\Sumo`) and put its `bin/` on `PATH`.
3. `pip install traci sumolib` (or use the copies bundled in `%SUMO_HOME%\tools`).

## Build the scenario

```bash
python sim/build_network.py
```

This runs `netconvert` on the node/edge files, then introspects the generated traffic light with
`sumolib` and writes `output/tls.add.xml` — a 5-phase program in the exact order the adapter maps:

| index | phase       | meaning              |
|-------|-------------|----------------------|
| 0     | `NS_GREEN`  | North/South green    |
| 1     | `NS_YELLOW` | North/South yellow   |
| 2     | `ALL_RED`   | clearance            |
| 3     | `EW_GREEN`  | East/West green      |
| 4     | `EW_YELLOW` | East/West yellow     |

Deriving the signal-state strings from the real controlled links (rather than hard-coding them)
keeps the phase indices valid no matter how `netconvert` numbers the connections.

## Drive it from Python

```python
from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import Direction
from sentinel.simulation.controllers import AdaptiveController
from sentinel.simulation.sumo import SumoTrafficEnvironment

env = SumoTrafficEnvironment(
    sumo_cfg="sim/output/demo.sumocfg",
    tls_id="C",
    approach_lanes={
        Direction.NORTH: ["N_in_0"], Direction.SOUTH: ["S_in_0"],
        Direction.EAST:  ["E_in_0"], Direction.WEST:  ["W_in_0"],
    },
    use_gui=True,  # sumo-gui for the visual demo
)
controller = AdaptiveController(DecisionSettings())

state = env.reset()
for _ in range(1800):
    state = env.step(controller.decide(state, dt=1.0))
print(env.metrics().avg_delay_s)
env.close()
```

Swap `AdaptiveController` for `FixedTimerController` to see the baseline, or run the whole A/B via
[`run_comparison`](../libs/sentinel/simulation/harness.py) with a SUMO env factory.

> **Note:** the SUMO path is integration-tested with the services in a later module (it needs the
> SUMO binary). The analytical twin gives identical, fully-tested closed-loop behaviour today —
> run `python -m sentinel.simulation.run_baseline`.
