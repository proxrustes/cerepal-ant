"use client";

import { Card, CardContent, Stack, Typography } from "@mui/material";
import { ReactNode } from "react";

type Props = {
  icon?: ReactNode;
  value: string;
  label: string;
  unit?: string;
};

export function TelemetryCard({ icon, value, label, unit }: Props) {
  return (
    <Card
      sx={{
        borderColor: "primary.main",
        borderWidth: 2,
        borderStyle: "solid",
        borderRadius: 1,
      }}
      elevation={0}
    >
      <CardContent>
        <Stack direction="row" spacing={2} alignItems="center">
          {icon}
          <Stack spacing={0.3}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {value}
              {unit ? ` ${unit}` : ""}
            </Typography>
            <Typography
              variant="caption"
              sx={{ textTransform: "uppercase", letterSpacing: 1 }}
            >
              {label}
            </Typography>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
