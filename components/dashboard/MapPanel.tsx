"use client";

import { Box, Typography } from "@mui/material";

export function MapPanel() {
  return (
    <Box
      sx={{
        borderRadius: 1,
        bgcolor: "primary.main",
        flex: 1,
        height: { xs: 320, md: 520 },
        position: "relative",
        overflow: "hidden",
      }}
    >
      <Typography
        variant="subtitle2"
        sx={{
          position: "absolute",
          top: 12,
          left: 16,
          opacity: 0.7,
        }}
      >
        MAP / ROUTE MOCK
      </Typography>
    </Box>
  );
}
