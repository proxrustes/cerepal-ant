"use client";

import { Box, Typography } from "@mui/material";

export function CameraPanel() {
  return (
    <Box
      sx={{
        flex: 1,
        borderColor: "primary.main",
        borderWidth: 2,
        borderStyle: "solid",
        borderRadius: 1,
        overflow: "hidden",
        position: "relative",
        bgcolor: "grey.800",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          background:
            "url(https://i.pinimg.com/736x/6c/f0/85/6cf08577b87e238ca4ea29614c11c66c.jpg) center/cover",
          filter: "saturate(1.1)",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          border: "2px solid rgba(255,255,255,0.5)",
          m: 2,
          borderRadius: 2,
        }}
      />
      {["N", "E", "S", "W"].map((dir) => (
        <Typography
          key={dir}
          variant="caption"
          sx={{
            position: "absolute",
            color: "white",
            fontWeight: 700,
            ...(dir === "N" && {
              top: 8,
              left: "50%",
              transform: "translateX(-50%)",
            }),
            ...(dir === "S" && {
              bottom: 8,
              left: "50%",
              transform: "translateX(-50%)",
            }),
            ...(dir === "E" && {
              right: 8,
              top: "50%",
              transform: "translateY(-50%)",
            }),
            ...(dir === "W" && {
              left: 8,
              top: "50%",
              transform: "translateY(-50%)",
            }),
          }}
        >
          {dir}
        </Typography>
      ))}
    </Box>
  );
}
