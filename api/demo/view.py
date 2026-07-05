import json
from datetime import datetime, timezone


def handler(request=None, context=None):
    payload = {
        "step_count": 3,
        "latest_state": {
            "intersection_id": "intersection-1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lanes": {
                "NORTH": {"vehicle_count": 8, "avg_wait_s": 14.2},
                "SOUTH": {"vehicle_count": 6, "avg_wait_s": 10.8},
                "EAST": {"vehicle_count": 3, "avg_wait_s": 4.5},
                "WEST": {"vehicle_count": 2, "avg_wait_s": 3.1},
            },
            "current_phase": "NS_GREEN",
        },
        "latest_decision": {
            "command": {
                "action": {"value": "KEEP_GREEN"},
                "reason_code": "current_axis_serving",
                "policy_version": "adaptive-lqf-v1",
            }
        },
        "latest_signal": {
            "phase": "NS_GREEN",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": "demo-view: step=3 phase=N-S GREEN vehicles=19 decision=KEEP_GREEN",
    }
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(payload),
    }
