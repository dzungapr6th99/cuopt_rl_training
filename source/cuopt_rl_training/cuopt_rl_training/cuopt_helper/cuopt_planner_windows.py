# cuopt_planner_windows_2510_less_calls.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import time
import json
import requests
import re

class CuOptPlanner:
    def __init__(
        self,
        time_limits: float = 5.0,
        seed: int = 0,
        server_base_url: str = "http://localhost:8888",
        # giảm tần số poll xuống 1s/lần
        poll_interval_s: float = 1.0,
        # tổng thời gian chờ tối đa cho 1 request (giây)
        max_wait_s: float = 60.0,
        http_timeout_s: float = 30.0,
    ):
        self.time_limits = float(time_limits)
        self.seed = int(seed)
        self.server_base_url = server_base_url.rstrip("/")
        self.poll_interval_s = float(poll_interval_s)
        self.max_wait_s = float(max_wait_s)
        self.http_timeout_s = float(http_timeout_s)

    def plan(
        self,
        cost_matrix: List[List[float]],
        orders: List[Dict[str, Any]],
        vehicle_starts: List[int],
        vehicle_returns: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        if vehicle_returns is None:
            vehicle_returns = list(vehicle_starts)

        n_locations = len(cost_matrix)
        if n_locations == 0 or any(len(r) != n_locations for r in cost_matrix):
            raise ValueError("cost_matrix must be NxN")

        payload: Dict[str, Any] = {
            "cost_matrix_data": {"data": {"0": cost_matrix}},
            "task_data": {"task_locations": [int(o["location"]) for o in orders]},
            "fleet_data": {
                "vehicle_locations": [[int(s), int(e)] for s, e in zip(vehicle_starts, vehicle_returns)]
            },
            "solver_config": {
                "time_limit": self.time_limits
            },
        }
        #print(payload)
        req_id = self._submit_request(payload)

        # ✅ poll status (1s/lần) và CHỈ KHI completed mới gọi /solution
        status = self._wait_until_completed(req_id)
        if status != "completed":
            raise RuntimeError(f"cuOpt returned unexpected status={status!r}, reqId={req_id}")

        sol = self._get_solution(req_id)
        return self._parse_solution_to_rows(sol)

    # -------------------------
    # HTTP (cuOpt 25.10)
    # -------------------------
    def _submit_request(self, payload: Dict[str, Any]) -> str:
        url = f"{self.server_base_url}/cuopt/request"
        r = requests.post(url, json=payload, timeout=self.http_timeout_s)
        r.raise_for_status()
        try:
            obj = r.json()
        except Exception as e:
            raise RuntimeError(
                f"Submit returned non-JSON. HTTP={r.status_code}, ct={r.headers.get('content-type')}, "
                f"body_head={r.text[:400]!r}"
            ) from e

        req_id = obj.get("reqId") or obj.get("reqID") or obj.get("requestId")
        if not req_id:
            raise RuntimeError(f"Submit JSON has no reqId field: {obj}")
        return str(req_id)

    def _wait_until_completed(self, req_id: str) -> str:
        url = f"{self.server_base_url}/cuopt/request/{req_id}"
        deadline = time.time() + self.max_wait_s
        last = ""
        print("wait for id:", req_id)
        while time.time() < deadline:
            r = requests.get(
                url,
                timeout=self.http_timeout_s,
                headers={"Accept": "text/plain, application/json"},
            )
            r.raise_for_status()


            text = self._normalize_status(r.content)
            last = text

            last = text
            print("Get cuopt result:", last)
            if text == "completed":
                return "completed"
            if text in ("failed", "error", "cancelled", "canceled"):
                return text

            time.sleep(self.poll_interval_s)

        raise TimeoutError(f"cuOpt wait timed out. reqId={req_id}, last_status={last!r}")

    def _get_solution(self, req_id: str) -> Dict[str, Any]:
        url = f"{self.server_base_url}/cuopt/solution/{req_id}"
        r = requests.get(url, timeout=self.http_timeout_s)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            # tolerate wrong CT but JSON body
            try:
                return json.loads(r.text)
            except Exception as e:
                raise RuntimeError(
                    f"Solution returned non-JSON. HTTP={r.status_code}, ct={r.headers.get('content-type')}, "
                    f"body_head={r.text[:800]!r}"
                ) from e

    def _parse_solution_to_rows(self, sol: Dict[str, Any]) -> List[Dict[str, Any]]:
        # match exactly the sample you posted
        node = sol
        if "response" in node:
            node = node["response"]
        solver = node.get("solver_response", node)

        status = int(solver.get("status", 1))
        if status != 0:
            raise RuntimeError(f"cuOpt solve failed: status={status}")

        vehicle_data = solver.get("vehicle_data", {})
        rows: List[Dict[str, Any]] = []

        for veh_id_str, vinfo in vehicle_data.items():
            veh_id = int(veh_id_str)
            route = vinfo.get("route", [])
            eta = vinfo.get("arrival_stamp", [])

            for i, loc in enumerate(route):
                rows.append(
                    {
                        "seq": int(i),
                        "vehicle": int(veh_id),
                        "location": int(loc),
                        "eta": float(eta[i]) if i < len(eta) else 0.0,
                    }
                )
        return rows

    def evaluate_plan(self, plan: List[Any], metrics: Optional[List[str]] = None) -> Dict[str, float]:
        return {}
    @staticmethod
    def _normalize_status(raw_bytes: bytes) -> str:
        """
        Turn any weird response (BOM, utf-16, binary-ish) into a clean ascii status.
        Expected outputs: 'pending'/'running'/'completed'/...
        """
        if not raw_bytes:
            return ""

        # Try common decodes
        text = ""
        for enc in ("utf-8-sig", "utf-16", "utf-16le", "utf-16be", "utf-8", "latin-1"):
            try:
                text = raw_bytes.decode(enc, errors="ignore")
                if text:
                    break
            except Exception:
                continue

        # Strip quotes, whitespace
        text = text.strip().strip('"').strip("'").lower()

        # Remove all non-letters (kills BOM leftovers / weird symbols like '�', NULs, etc.)
        text = re.sub(r"[^a-z]+", "", text)

        return text
    
