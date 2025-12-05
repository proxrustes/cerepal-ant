"use client";

import * as React from "react";
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
} from "@mui/material";
import LayersIcon from "@mui/icons-material/Layers";
import MapIcon from "@mui/icons-material/Map";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import SatelliteAltIcon from "@mui/icons-material/SatelliteAlt";
import CheckIcon from "@mui/icons-material/Check";
import { MapStyleKey, TILE_LAYERS } from "./styles";

type Props = {
  value: MapStyleKey;
  onChange: (style: MapStyleKey) => void;
};

function mapStyleIcon(key: MapStyleKey) {
  if (key === "cartoDark") return <DarkModeIcon fontSize="small" />;
  if (key === "esriImagery") return <SatelliteAltIcon fontSize="small" />;
  return <MapIcon fontSize="small" />;
}

export function MapStyleControl({ value, onChange }: Props) {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleOpenMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseMenu = () => setAnchorEl(null);

  const handleSelectStyle = (key: MapStyleKey) => {
    onChange(key);
    setAnchorEl(null);
  };

  return (
    <Box>
      <IconButton
        size="small"
        onClick={handleOpenMenu}
        sx={{
          bgcolor: "rgba(15,23,42,0.9)",
          border: "1px solid",
          borderColor: "divider",
          "&:hover": { bgcolor: "rgba(15,23,42,1)" },
        }}
      >
        <LayersIcon fontSize="small" />
      </IconButton>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleCloseMenu}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        PaperProps={{
          sx: {
            bgcolor: "background.paper",
            minWidth: 220,
          },
        }}
      >
        <Box sx={{ px: 2, pt: 1, pb: 0.5 }}>
          <Typography variant="caption" sx={{ opacity: 0.7 }}>
            Kartenstil
          </Typography>
        </Box>

        {Object.entries(TILE_LAYERS).map(([key, cfg]) => {
          const k = key as MapStyleKey;
          const selected = k === value;
          return (
            <MenuItem
              key={key}
              selected={selected}
              onClick={() => handleSelectStyle(k)}
            >
              <ListItemIcon>{mapStyleIcon(k)}</ListItemIcon>
              <ListItemText primary={cfg.label} />
              {selected && (
                <ListItemIcon sx={{ minWidth: 24 }}>
                  <CheckIcon fontSize="small" />
                </ListItemIcon>
              )}
            </MenuItem>
          );
        })}
      </Menu>
    </Box>
  );
}
