"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AXIS = "#94A3B8";
const GRID = "#334155";

const tooltipStyle = {
  backgroundColor: "#1E293B",
  border: "1px solid #334155",
  borderRadius: 8,
  color: "#F8FAFC",
  fontSize: 12,
};

export function DailyVisitsArea({ data }: { data: { day: string; visits: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="visitsFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.5} />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="day" stroke={AXIS} fontSize={11} tickLine={false} />
        <YAxis stroke={AXIS} fontSize={11} tickLine={false} allowDecimals={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <Area
          type="monotone"
          dataKey="visits"
          stroke="#3B82F6"
          strokeWidth={2}
          fill="url(#visitsFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

const DONUT_COLORS = ["#3B82F6", "#8B5CF6"];

export function NewVsReturningDonut({
  newCount,
  returningCount,
}: {
  newCount: number;
  returningCount: number;
}) {
  const data = [
    { name: "New", value: newCount },
    { name: "Returning", value: returningCount },
  ];
  const total = newCount + returningCount;
  if (total === 0) {
    return (
      <div className="flex h-[260px] items-center justify-center text-sm text-text-secondary">
        No data
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={260}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          innerRadius={60}
          outerRadius={90}
          paddingAngle={2}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={DONUT_COLORS[i]} />
          ))}
        </Pie>
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 12, color: AXIS }} />
      </PieChart>
    </ResponsiveContainer>
  );
}

export function HourlyStackedBar({
  data,
}: {
  data: { hour: number; new: number; returning: number }[];
}) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <XAxis
          dataKey="hour"
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          tickFormatter={(h) => `${h}h`}
        />
        <YAxis stroke={AXIS} fontSize={11} tickLine={false} allowDecimals={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 12, color: AXIS }} />
        <Bar dataKey="returning" stackId="a" fill="#8B5CF6" radius={[0, 0, 0, 0]} />
        <Bar dataKey="new" stackId="a" fill="#3B82F6" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function MonthlyBar({ data }: { data: { label: string; value: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <XAxis dataKey="label" stroke={AXIS} fontSize={11} tickLine={false} />
        <YAxis stroke={AXIS} fontSize={11} tickLine={false} allowDecimals={false} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: GRID, opacity: 0.3 }} />
        <Bar dataKey="value" fill="#3B82F6" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DetectionQualityBar({
  high,
  medium,
  low,
}: {
  high: number;
  medium: number;
  low: number;
}) {
  const total = high + medium + low;
  if (total === 0) {
    return (
      <div className="flex h-[120px] items-center justify-center text-sm text-text-secondary">
        No detections in range
      </div>
    );
  }
  const seg = [
    { label: "High", value: high, color: "#10B981" },
    { label: "Medium", value: medium, color: "#F59E0B" },
    { label: "Low", value: low, color: "#EF4444" },
  ];
  return (
    <div className="space-y-4 py-2">
      <div className="flex h-4 w-full overflow-hidden rounded-full bg-card/40">
        {seg.map((s) => (
          <div
            key={s.label}
            style={{ width: `${(s.value / total) * 100}%`, backgroundColor: s.color }}
            className="h-full transition-all"
          />
        ))}
      </div>
      <div className="grid grid-cols-3 gap-2">
        {seg.map((s) => (
          <div key={s.label} className="text-center">
            <div className="flex items-center justify-center gap-1.5 text-xs text-text-secondary">
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              {s.label}
            </div>
            <p className="mt-1 text-lg font-semibold text-text-primary">
              {Math.round((s.value / total) * 100)}%
            </p>
            <p className="text-[11px] text-text-muted">{s.value.toLocaleString()}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export function FrequencyBar({ data }: { data: { bucket: string; count: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 0, right: 20, left: 10, bottom: 0 }}
      >
        <XAxis type="number" stroke={AXIS} fontSize={11} tickLine={false} allowDecimals={false} />
        <YAxis
          type="category"
          dataKey="bucket"
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          width={70}
        />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: GRID, opacity: 0.3 }} />
        <Bar dataKey="count" fill="#10B981" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
