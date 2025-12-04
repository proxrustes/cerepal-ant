"use client";

import { Box, Typography } from "@mui/material";

type Props = {
  coords: string;
};

export function CoordinateBar({ coords }: Props) {
  return (
    <Box
      sx={{
        borderRadius: 1,
        px: 4,
        py: 2,
        mb: 3,
        borderColor: "secondary.main",
        borderWidth: 2,
        borderStyle: "solid",
        display: "inline-block",
      }}
    >
      <Typography
        variant="h5"
        component="div"
        sx={{ fontWeight: 600, letterSpacing: 1 }}
      >
        {coords}
      </Typography>
    </Box>
  );
}
