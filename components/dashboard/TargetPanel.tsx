"use client";

import { useEffect, useState } from "react";
import {
  Box,
  Button,
  Icon,
  IconButton,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import RoomIcon from "@mui/icons-material/Room";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import { useTracking } from "../../context/TrackingContext";

import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";

export function TargetPanel() {
  const {
    target,
    beginTargetSelection,
    isPickingTarget,
    setTargetCoords,
    focusOnTarget,
  } = useTracking();

  const [latInput, setLatInput] = useState(target.lat.toFixed(6));
  const [lonInput, setLonInput] = useState(target.lon.toFixed(6));

  useEffect(() => {
    setLatInput(target.lat.toFixed(6));
    setLonInput(target.lon.toFixed(6));
  }, [target.lat, target.lon]);

  const applyManual = () => {
    const lat = parseFloat(latInput.replace(",", "."));
    const lon = parseFloat(lonInput.replace(",", "."));
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return;
    }
    setTargetCoords({ lat, lon });
  };

  return (
    <Box
      sx={{
        borderRadius: 1,
        px: 3,
        py: 2,
        backgroundColor: "background.paper",
        mb: 3,
      }}
    >
      <Stack spacing={2}>
        <Stack
          direction="row"
          spacing={1}
          alignItems="center"
          justifyContent={"space-between"}
        >
          <Stack direction={"row"} alignItems={"center"} spacing={1}>
            <RoomIcon />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Ziel: {target.lat.toFixed(5)}, {target.lon.toFixed(5)}
            </Typography>
            <IconButton onClick={focusOnTarget}>
              <CenterFocusStrongIcon />
            </IconButton>
          </Stack>

          <Stack direction={"row"} gap={2}>
            <TextField
              label="Latitude (lat)"
              size="small"
              value={latInput}
              onChange={(e) => setLatInput(e.target.value)}
            />
            <TextField
              label="Longitude (lon)"
              size="small"
              value={lonInput}
              onChange={(e) => setLonInput(e.target.value)}
            />
          </Stack>
          <Stack direction={"row"} alignItems={"center"} spacing={1}>
            <Button variant="outlined" onClick={applyManual}>
              Manuelle Eingabe
            </Button>
            <Button
              variant="contained"
              startIcon={<MyLocationIcon />}
              color={isPickingTarget ? "warning" : "primary"}
              onClick={beginTargetSelection}
              sx={{ width: 320 }}
            >
              {isPickingTarget
                ? " Klicken auf die Karte"
                : "Setze Ziel auf der Karte"}
            </Button>
          </Stack>
        </Stack>
      </Stack>
    </Box>
  );
}
