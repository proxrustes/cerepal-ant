"use client";

import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  IconButton,
} from "@mui/material";
import AndroidIcon from "@mui/icons-material/Android";
import DirectionsRunIcon from "@mui/icons-material/DirectionsRun";
import BatteryChargingFullIcon from "@mui/icons-material/BatteryChargingFull";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import { Robot, useTracking } from "../../context/TrackingContext";
import { bearingDeg, toCardinalDE } from "../../lib/mapUtils";

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
  const {
    robots,
    base,
    target,
    selectedRobotIds,
    toggleRobotSelection,
    focusOnRobot,
  } = useTracking();

  const handleRowClick = (id: string) => {
    toggleRobotSelection(id);
  };

  return (
    <Box
      sx={{
        mb: 2,
        borderColor: "secondary.main",
        borderWidth: 2,
        borderStyle: "solid",
        borderRadius: 1,
        p: 1.5,
      }}
    >
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 600 }}>Name</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Koordinaten</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Geschwindigkeit</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Richtung</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Akkustand</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Aufgabe</TableCell>
            <TableCell sx={{ fontWeight: 600 }} align="center">
              Zentrieren
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
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
            const isSelected = selectedRobotIds.includes(r.id);

            return (
              <TableRow
                key={r.id}
                hover
                selected={isSelected}
                onClick={() => handleRowClick(r.id)}
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

                <TableCell align="center">
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      focusOnRobot(r.id);
                    }}
                  >
                    <CenterFocusStrongIcon fontSize="small" />
                  </IconButton>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Box>
  );
}
