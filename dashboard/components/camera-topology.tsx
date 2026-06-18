"use client";

import { useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { Plus, Trash2, ArrowLeftRight } from "lucide-react";

import { api, fetcher } from "@/lib/api";
import {
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  Input,
  Select,
  Toggle,
} from "@/components/ui";

interface TopologyRow {
  id: string;
  camera_a: string;
  camera_b: string;
  min_travel_seconds: number | null;
  max_expected_seconds: number | null;
  transition_enabled: boolean;
}

export function CameraTopologyManager() {
  const { mutate } = useSWRConfig();
  const { data: rows, error } = useSWR<TopologyRow[]>("admin/camera-topology", fetcher);
  const { data: cameras } = useSWR<string[]>("admin/cameras", fetcher);

  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [minT, setMinT] = useState("");
  const [maxT, setMaxT] = useState("");
  const [enabled, setEnabled] = useState(true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const camOptions = (cameras ?? []).map((c) => ({ value: c, label: c }));

  async function add() {
    if (!a || !b || a === b) {
      setMsg("Pick two different cameras.");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      await api.post("admin/camera-topology", {
        camera_a: a,
        camera_b: b,
        min_travel_seconds: minT ? Number(minT) : null,
        max_expected_seconds: maxT ? Number(maxT) : null,
        transition_enabled: enabled,
      });
      await mutate("admin/camera-topology");
      setMinT("");
      setMaxT("");
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    try {
      await api.del(`admin/camera-topology/${id}`);
      await mutate("admin/camera-topology");
    } catch (e) {
      setMsg((e as Error).message);
    }
  }

  return (
    <Card>
      <CardTitle
        action={
          <Badge tone="neutral">
            <ArrowLeftRight className="h-3 w-3" /> Cross-camera
          </Badge>
        }
      >
        Camera Topology
      </CardTitle>
      <p className="mb-4 text-sm text-text-secondary">
        Travel-time constraints between cameras. Cross-camera matching rejects
        transitions faster than the minimum and de-prioritises disabled pairs.
        Only used when <code className="text-primary">CROSS_CAMERA_ENABLED</code> is on.
      </p>

      {/* Add form */}
      <div className="mb-4 grid grid-cols-1 gap-2 sm:grid-cols-6">
        <div className="sm:col-span-1">
          {camOptions.length > 0 ? (
            <Select value={a} options={[{ value: "", label: "Camera A…" }, ...camOptions]} onChange={setA} />
          ) : (
            <Input value={a} onChange={setA} placeholder="Camera A" />
          )}
        </div>
        <div className="sm:col-span-1">
          {camOptions.length > 0 ? (
            <Select value={b} options={[{ value: "", label: "Camera B…" }, ...camOptions]} onChange={setB} />
          ) : (
            <Input value={b} onChange={setB} placeholder="Camera B" />
          )}
        </div>
        <Input value={minT} onChange={setMinT} type="number" placeholder="Min s" />
        <Input value={maxT} onChange={setMaxT} type="number" placeholder="Max s" />
        <div className="flex items-center gap-2">
          <Toggle checked={enabled} onChange={setEnabled} label="Enabled" />
        </div>
        <Button onClick={add} disabled={busy}>
          <Plus className="h-4 w-4" /> Add
        </Button>
      </div>
      {msg && <p className="mb-3 text-sm text-danger">{msg}</p>}

      {error ? (
        <p className="text-sm text-text-muted">
          Topology unavailable (run migration 011 + set ADMIN_API_KEY).
        </p>
      ) : !rows || rows.length === 0 ? (
        <EmptyState message="No camera transitions configured yet." />
      ) : (
        <ul className="divide-y divide-card/40 text-sm">
          {rows.map((r) => (
            <li key={r.id} className="flex items-center justify-between gap-3 py-2.5">
              <div className="flex items-center gap-2">
                <span className="font-medium text-text-primary">{r.camera_a}</span>
                <ArrowLeftRight className="h-3.5 w-3.5 text-text-muted" />
                <span className="font-medium text-text-primary">{r.camera_b}</span>
                {!r.transition_enabled && <Badge tone="danger">blocked</Badge>}
              </div>
              <div className="flex items-center gap-4">
                <span className="text-xs text-text-secondary">
                  {r.min_travel_seconds ?? "—"}s – {r.max_expected_seconds ?? "—"}s
                </span>
                <Button variant="ghost" size="sm" onClick={() => remove(r.id)}>
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
