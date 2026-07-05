"""Build the SUMO network + scenario for the Sentinel demo.

Runs ``netconvert`` on the hand-written node/edge files, then introspects the generated
traffic-light with ``sumolib`` to emit a TLS program whose phase order **exactly matches**
``sentinel.simulation.sumo._PHASE_TO_INDEX`` (NS green, NS yellow, all-red, EW green, EW yellow).
Doing it this way -- deriving the signal state strings from the real controlled links rather than
hard-coding them -- keeps the adapter's phase indices valid regardless of how netconvert numbers
the connections.

Requires a SUMO install (``SUMO_HOME`` set, ``netconvert`` on PATH). Run once before using
:class:`~sentinel.simulation.sumo.SumoTrafficEnvironment`::

    python sim/build_network.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_NET = _HERE / "network"
_OUT = _HERE / "output"


def _require_sumo() -> None:
    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME is not set. Install SUMO and set SUMO_HOME. See sim/README.md.")
    tools = Path(os.environ["SUMO_HOME"]) / "tools"
    if str(tools) not in sys.path:
        sys.path.append(str(tools))


def _run_netconvert(net_file: Path) -> None:
    cmd = [
        "netconvert",
        "--node-files", str(_NET / "intersection.nod.xml"),
        "--edge-files", str(_NET / "intersection.edg.xml"),
        "--output-file", str(net_file),
        "--tls.guess", "true",
    ]
    subprocess.run(cmd, check=True)


def _direction_of_edge(edge_id: str) -> str:
    """Return the approach letter (N/S/E/W) an inbound edge id belongs to."""
    return edge_id.split("_")[0]


def _write_tls_program(net_file: Path, tls_add: Path, *, yellow_s: int = 3, all_red_s: int = 2) -> None:
    import sumolib  # provided by SUMO_HOME/tools

    net = sumolib.net.readNet(str(net_file), withPrograms=True)
    tls = net.getTrafficLights()[0]
    tls_id = tls.getID()
    connections = tls.getConnections()  # list of (inLane, outLane, linkIndex)
    n_links = len({link_index for _, _, link_index in connections})

    # Classify each controlled link as North/South axis or East/West axis.
    ns_links: set[int] = set()
    for in_lane, _out_lane, link_index in connections:
        edge_id = in_lane.getEdge().getID()
        if _direction_of_edge(edge_id) in ("N", "S"):
            ns_links.add(link_index)

    def state(green_axis: str | None, *, yellow: bool = False) -> str:
        chars = []
        for i in range(n_links):
            is_ns = i in ns_links
            active = (green_axis == "NS" and is_ns) or (green_axis == "EW" and not is_ns)
            if not active:
                chars.append("r")
            else:
                chars.append("y" if yellow else "G")
        return "".join(chars)

    phases = [
        ('99', state("NS")),                      # 0 NS_GREEN (duration overridden via TraCI)
        (str(yellow_s), state("NS", yellow=True)),  # 1 NS_YELLOW
        (str(all_red_s), state(None)),             # 2 ALL_RED
        ('99', state("EW")),                       # 3 EW_GREEN
        (str(yellow_s), state("EW", yellow=True)),  # 4 EW_YELLOW
    ]
    phase_xml = "\n".join(
        f'        <phase duration="{dur}" state="{st}"/>' for dur, st in phases
    )
    tls_add.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<additional>\n"
        f'    <tlLogic id="{tls_id}" type="static" programID="sentinel" offset="0">\n'
        f"{phase_xml}\n"
        "    </tlLogic>\n"
        "</additional>\n",
        encoding="utf-8",
    )
    print(f"Wrote TLS program for '{tls_id}' with {n_links} controlled links -> {tls_add}")


def _write_sumocfg(cfg: Path, net_file: Path, tls_add: Path) -> None:
    cfg.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<configuration>\n"
        "    <input>\n"
        f'        <net-file value="{net_file.name}"/>\n'
        f'        <route-files value="{(_NET / "routes.rou.xml").resolve()}"/>\n'
        f'        <additional-files value="{tls_add.name}"/>\n'
        "    </input>\n"
        "    <time>\n"
        '        <begin value="0"/>\n'
        '        <end value="3600"/>\n'
        "    </time>\n"
        "</configuration>\n",
        encoding="utf-8",
    )
    print(f"Wrote scenario config -> {cfg}")


def main() -> None:
    _require_sumo()
    _OUT.mkdir(exist_ok=True)
    net_file = _OUT / "intersection.net.xml"
    tls_add = _OUT / "tls.add.xml"
    cfg = _OUT / "demo.sumocfg"

    _run_netconvert(net_file)
    _write_tls_program(net_file, tls_add)
    _write_sumocfg(cfg, net_file, tls_add)
    print("\nSUMO scenario ready. Run the demo with SumoTrafficEnvironment(sumo_cfg=str(cfg), ...).")


if __name__ == "__main__":
    main()
