// src/components/dashboard/TargetPanel.tsx
"use client";

import { useEffect, useState } from "react";
import { Box, Button, Stack, TextField, Typography } from "@mui/material";
import RoomIcon from "@mui/icons-material/Room";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import { useTracking } from "../../context/TrackingContext";

export function TargetPanel() {
  const { target, beginTargetSelection, isPickingTarget, setTargetCoords } =
    useTracking();

  const [latInput, setLatInput] = useState(target.lat.toFixed(6));
  const [lonInput, setLonInput] = useState(target.lon.toFixed(6));

  // если цель изменилась (например, по клику на карте) — обновляем инпуты
  useEffect(() => {
    setLatInput(target.lat.toFixed(6));
    setLonInput(target.lon.toFixed(6));
  }, [target.lat, target.lon]);

  const applyManual = () => {
    const lat = parseFloat(latInput.replace(",", "."));
    const lon = parseFloat(lonInput.replace(",", "."));
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      // тут можно повесить валидацию/тост, пока просто ничего не делаем
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
        mb: 3,
        borderColor: "primary.main",
        borderWidth: 2,
        borderStyle: "solid",
      }}
    >
      <Stack spacing={2}>
        <Stack direction="row" spacing={1} alignItems="center">
          <RoomIcon />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Ціль
          </Typography>
        </Stack>

        <Typography variant="body2" sx={{ opacity: 0.8 }}>
          Поточні координати:&nbsp;
          <strong>
            {target.lat.toFixed(5)}, {target.lon.toFixed(5)}
          </strong>
        </Typography>

        <Stack
          direction={{ xs: "column", sm: "row" }}
          spacing={2}
          alignItems={{ xs: "stretch", sm: "flex-end" }}
        >
          <TextField
            label="Широта (lat)"
            size="small"
            value={latInput}
            onChange={(e) => setLatInput(e.target.value)}
            fullWidth
          />
          <TextField
            label="Довгота (lon)"
            size="small"
            value={lonInput}
            onChange={(e) => setLonInput(e.target.value)}
            fullWidth
          />
          <Button variant="outlined" onClick={applyManual}>
            Задать цель вручную
          </Button>
        </Stack>

        <Stack direction="row" spacing={2}>
          <Button
            variant="contained"
            startIcon={<MyLocationIcon />}
            color={isPickingTarget ? "warning" : "primary"}
            onClick={beginTargetSelection}
          >
            Задать цель на карте
          </Button>

          {isPickingTarget && (
            <Typography variant="caption" sx={{ opacity: 0.8 }}>
              Кликни по карте, чтобы выбрать новую точку.
            </Typography>
          )}
        </Stack>
      </Stack>
    </Box>
  );
}
