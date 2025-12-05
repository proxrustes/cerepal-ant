"use client";

import { Stack, Button } from "@mui/material";
import { useTracking } from "../../context/TrackingContext";
import UndoIcon from "@mui/icons-material/Undo";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import WhatshotIcon from "@mui/icons-material/Whatshot";

export function ControlButtonsBar() {
  const { commandToBase, commandToTarget } = useTracking();
  return (
    <Stack sx={{ gap: 2 }} direction="row" justifyContent="center">
      <Button
        variant="contained"
        color="primary"
        onClick={commandToTarget}
        fullWidth
        sx={{ height: 80, fontSize: 28, fontWeight: 700, borderRadius: 1 }}
        startIcon={<MyLocationIcon />}
      >
        ZIEL ERFASSEN
      </Button>
      <Button
        variant="contained"
        color="secondary"
        fullWidth
        onClick={commandToBase}
        sx={{ height: 80, fontSize: 28, fontWeight: 700, borderRadius: 1 }}
        startIcon={<UndoIcon />}
      >
        ZUR BASIS
      </Button>
      <Button
        variant="contained"
        disabled
        color="error"
        fullWidth
        sx={{ height: 80, fontSize: 28, fontWeight: 700, borderRadius: 1 }}
        startIcon={<WhatshotIcon />}
      >
        SELBSTZERSTÖRUNG
      </Button>
    </Stack>
  );
}
