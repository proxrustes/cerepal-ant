// src/components/dashboard/RobotSelector.tsx
"use client";

import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from "@mui/material";
import AndroidIcon from "@mui/icons-material/Android";
import GroupsIcon from "@mui/icons-material/Groups";
import DirectionsRunIcon from "@mui/icons-material/DirectionsRun";
import BatteryChargingFullIcon from "@mui/icons-material/BatteryChargingFull";
import { Robot, useTracking } from "../../context/TrackingContext";

function taskLabel(mode: "idle" | "toBase" | "toTarget"): string {
  switch (mode) {
    case "toBase":
      return "Aufgabe: Zur Basis";
    case "toTarget":
      return "Aufgabe: Zum Ziel";
    case "idle":
    default:
      return "Aufgabe: Warten";
  }
}

// азимут → направление
function bearingDeg(
  from: { lat: number; lon: number },
  to: { lat: number; lon: number }
): number {
  const φ1 = (from.lat * Math.PI) / 180;
  const φ2 = (to.lat * Math.PI) / 180;
  const λ1 = (from.lon * Math.PI) / 180;
  const λ2 = (to.lon * Math.PI) / 180;
  const y = Math.sin(λ2 - λ1) * Math.cos(φ2);
  const x =
    Math.cos(φ1) * Math.sin(φ2) -
    Math.sin(φ1) * Math.cos(φ2) * Math.cos(λ2 - λ1);
  const θ = Math.atan2(y, x);
  return (θ * 180) / Math.PI;
}

function toCardinalDE(deg: number): string {
  const d = (deg + 360) % 360;
  const dirs = ["N", "NO", "O", "SO", "S", "SW", "W", "NW"];
  const idx = Math.round(d / 45) % 8;
  return dirs[idx];
}

function directionLabel(
  r: Robot,
  base: { lat: number; lon: number },
  target: { lat: number; lon: number }
): string {
  let dest: { lat: number; lon: number } | null = null;
  if (r.mode === "toBase") dest = base;
  else if (r.mode === "toTarget") dest = target;
  else return "–";

  const deg = bearingDeg(r.position, dest);
  return toCardinalDE(deg);
}

export function RobotSelector() {
  const { robots, selected, setSelected, base, target } = useTracking();

  const handleSelect = (value: string) => {
    setSelected(value as any); // "all" | robotId
  };

  return (
    <Box
      sx={{
        mb: 2,
        borderColor: "primary.main",
        borderWidth: 2,
        borderStyle: "solid",
        borderRadius: 1,
        p: 1.5,
      }}
    >
      <Typography variant="subtitle2" sx={{ opacity: 0.8, mb: 1 }}>
        Roboter
      </Typography>

      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 600 }}>Name</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Koordinaten</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Geschwindigkeit</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Richtung</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Akkustand</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Aufgabe</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {/* строка "все роботы" */}
          <TableRow
            hover
            selected={selected === "all"}
            onClick={() => handleSelect("all")}
            sx={{ cursor: "pointer" }}
          >
            <TableCell>
              <Box display="flex" alignItems="center" gap={1}>
                <GroupsIcon fontSize="small" />
                <Typography variant="body2">Alle Roboter</Typography>
              </Box>
            </TableCell>
            <TableCell>–</TableCell>
            <TableCell>–</TableCell>
            <TableCell>–</TableCell>
            <TableCell>–</TableCell>
            <TableCell>
              <Typography variant="caption">
                Befehl gilt für alle ({robots.length})
              </Typography>
            </TableCell>
          </TableRow>

          {/* отдельные роботы */}
          {robots.map((r) => {
            const lat = r.position.lat.toFixed(4);
            const lon = r.position.lon.toFixed(4);
            const speedKmh = r.mode === "idle" ? 0 : 30;
            const dir = directionLabel(r, base, target);
            const battery = Math.round(r.battery);
            const batteryColor =
              battery > 60
                ? "success.main"
                : battery > 30
                ? "warning.main"
                : "error.main";

            return (
              <TableRow
                key={r.id}
                hover
                selected={selected === r.id}
                onClick={() => handleSelect(r.id)}
                sx={{ cursor: "pointer" }}
              >
                <TableCell>
                  <Box display="flex" alignItems="center" gap={1}>
                    <AndroidIcon fontSize="small" />
                    <Typography variant="body2" fontWeight={600}>
                      {r.name}
                    </Typography>
                  </Box>
                </TableCell>

                <TableCell>
                  <Typography variant="caption">
                    {lat}, {lon}
                  </Typography>
                </TableCell>

                <TableCell>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <DirectionsRunIcon fontSize="inherit" />
                    <Typography variant="caption">{speedKmh} km/h</Typography>
                  </Box>
                </TableCell>

                <TableCell>
                  <Typography variant="caption">{dir}</Typography>
                </TableCell>

                <TableCell>
                  <Box
                    display="flex"
                    alignItems="center"
                    gap={0.5}
                    sx={{ color: batteryColor }}
                  >
                    <BatteryChargingFullIcon fontSize="inherit" />
                    <Typography variant="caption">{battery} %</Typography>
                  </Box>
                </TableCell>

                <TableCell>
                  <Typography variant="caption">{taskLabel(r.mode)}</Typography>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Box>
  );
}
