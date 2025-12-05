"use client";

import { Box, Grid, Stack } from "@mui/material";
import { MapPanel } from "@/components/map/MapPanel";
import { CameraPanel } from "@/components/dashboard/CameraPanel";
import { ControlButtonsBar } from "@/components/dashboard/ControlButtonsBar";
import { RobotSelector } from "../components/dashboard/RobotSelector";
import { TargetPanel } from "../components/dashboard/TargetPanel";

export default function DashboardPage() {
  return (
    <Stack spacing={4} sx={{ p: 4 }}>
      <TargetPanel />

      <Grid container spacing={3}>
        <Grid size={8}>
          <MapPanel />
        </Grid>

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
          </Box>
        </Grid>
      </Grid>
      <RobotSelector />
      <ControlButtonsBar />
    </Stack>
  );
}
