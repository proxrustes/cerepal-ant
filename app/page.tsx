"use client";

import { Box, Grid } from "@mui/material";
import { CoordinateBar } from "@/components/dashboard/CoordinateBar";
import { MapPanel } from "@/components/dashboard/MapPanel";
import { CameraPanel } from "@/components/dashboard/CameraPanel";
import { TelemetryCard } from "@/components/dashboard/TelemetryCard";
import { ControlButtonsBar } from "@/components/dashboard/ControlButtonsBar";
import { MOCK_TELEMETRY } from "@/mock/telemetry";
import SpeedIcon from "@mui/icons-material/Speed";
import BatteryChargingFullIcon from "@mui/icons-material/BatteryChargingFull";
import NavigationIcon from "@mui/icons-material/Navigation";
import HeightIcon from "@mui/icons-material/Height";

export default function DashboardPage() {
  const t = MOCK_TELEMETRY;

  return (
    <Box
      sx={{
        minHeight: "100vh",
        bgcolor: "background.default",
        color: "text.primary",
        p: { xs: 2, md: 3 },
      }}
    >
      <CoordinateBar coords={t.coords} />

      <Grid container spacing={3}>
        {/* Map */}
        <Grid size={8}>
          <MapPanel />
        </Grid>

        {/* Right column: camera + telemetry */}
        <Grid size={4}>
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              gap: 2.5,
              height: "100%",
            }}
          >
            <CameraPanel />

            <Grid container spacing={2}>
              <Grid size={6}>
                <TelemetryCard
                  icon={<SpeedIcon />}
                  value={t.speedKmh.toString()}
                  unit="км/год"
                  label="швидкість"
                />
              </Grid>
              <Grid size={6}>
                <TelemetryCard
                  icon={<BatteryChargingFullIcon />}
                  value={t.battery.toString()}
                  unit="%"
                  label="заряд"
                />
              </Grid>
              <Grid size={6}>
                <TelemetryCard
                  icon={<NavigationIcon />}
                  value={t.heading}
                  label="напрям"
                />
              </Grid>
              <Grid size={6}>
                <TelemetryCard
                  icon={<HeightIcon />}
                  value={t.altitude.toString()}
                  unit="м"
                  label="висота"
                />
              </Grid>
            </Grid>
          </Box>
        </Grid>
      </Grid>

      <ControlButtonsBar />
    </Box>
  );
}
