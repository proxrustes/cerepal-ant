"use client";

import { Stack, Button } from "@mui/material";
import { useTracking } from "../../context/TrackingContext";

export function ControlButtonsBar() {
  const { commandToBase, commandToTarget } = useTracking();
  return (
    <Stack sx={{ gap: 2 }} direction="row" justifyContent="center">
      <Button
        variant="contained"
        color="success"
        onClick={commandToTarget}
        fullWidth
        sx={{ height: 80, fontSize: 28, fontWeight: 700 }}
      >
        ZIEL ERFASSEN
      </Button>
      <Button
        variant="contained"
        color="warning"
        fullWidth
        onClick={commandToBase}
        sx={{ height: 80, fontSize: 28, fontWeight: 700 }}
      >
        ZUR BASIS
      </Button>
      <Button
        variant="contained"
        disabled
        color="error"
        fullWidth
        sx={{ height: 80, fontSize: 28, fontWeight: 700 }}
      >
        SELBSTZERSTÖRUNG
      </Button>
    </Stack>
  );
}
