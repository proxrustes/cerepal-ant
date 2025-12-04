"use client";

import { Box, Button } from "@mui/material";

export function ControlButtonsBar() {
  return (
    <Box
      sx={{
        mt: 3,
        display: "grid",
        gap: 2,
        gridTemplateColumns: { xs: "1fr", md: "1.4fr 1fr 1.4fr" },
      }}
    >
      <Button
        variant="contained"
        color="success"
        sx={{
          py: 2,
          fontSize: 18,
          fontWeight: 700,
          borderRadius: 2,
        }}
      >
        ЗАХОПЛЕННЯ ЦІЛІ
      </Button>
      <Button
        variant="contained"
        color="warning"
        sx={{
          py: 2,
          fontSize: 18,
          fontWeight: 700,
          borderRadius: 2,
        }}
      >
        НА БАЗУ
      </Button>
      <Button
        variant="outlined"
        color="error"
        sx={{
          py: 2,
          fontSize: 18,
          fontWeight: 700,
          borderRadius: 2,
          borderWidth: 2,
        }}
      >
        САМОЗНИЩЕННЯ
      </Button>
    </Box>
  );
}
