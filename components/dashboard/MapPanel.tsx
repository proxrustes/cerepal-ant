// src/components/dashboard/MapPanel.tsx
"use client";

import { Avatar, Box } from "@mui/material";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMapEvents,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import HomeIcon from "@mui/icons-material/Home";
import FlagIcon from "@mui/icons-material/Flag";
import PestControlIcon from "@mui/icons-material/PestControl";
import { renderToString } from "react-dom/server";
import { useTracking } from "../../context/TrackingContext";
import { useEffect } from "react";

const robotIcon = L.divIcon({
  html: renderToString(
    <PestControlIcon style={{ fontSize: 42, color: "black" }} />
  ),
  className: "mui-leaflet-marker",
  iconSize: [30, 30],
  iconAnchor: [15, 30],
});

const robotSelectedIcon = L.divIcon({
  html: renderToString(
    <PestControlIcon style={{ fontSize: 42, color: "orange" }} />
  ),
  className: "mui-leaflet-marker",
  iconSize: [30, 30],
  iconAnchor: [15, 30],
});

const baseIcon = L.divIcon({
  html: renderToString(<HomeIcon style={{ fontSize: 28, color: "green" }} />),
  className: "mui-leaflet-marker",
  iconSize: [28, 28],
  iconAnchor: [14, 28],
});

const targetIcon = L.divIcon({
  html: renderToString(<FlagIcon style={{ fontSize: 28, color: "red" }} />),
  className: "mui-leaflet-marker",
  iconSize: [28, 28],
  iconAnchor: [14, 28],
});

function TargetClickHandler({
  enabled,
  onSelect,
}: {
  enabled: boolean;
  onSelect: (lat: number, lon: number) => void;
}) {
  useMapEvents({
    click(e) {
      if (!enabled) return;
      onSelect(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function MapFocusController() {
  const { mapFocus, robots, target } = useTracking();
  const map = useMap();

  useEffect(() => {
    if (!mapFocus) return;

    let lat: number | null = null;
    let lon: number | null = null;

    if (mapFocus.type === "target") {
      lat = target.lat;
      lon = target.lon;
    } else {
      const r = robots.find((x) => x.id === mapFocus.id);
      if (!r) return;
      lat = r.position.lat;
      lon = r.position.lon;
    }

    map.setView([lat, lon], map.getZoom(), { animate: true });
  }, [mapFocus, robots, target, map]);

  return null;
}

export function MapPanel() {
  const {
    robots,
    base,
    target,
    selectedRobotIds,
    isPickingTarget,
    setTargetCoords,
  } = useTracking();

  const center: [number, number] = [base.lat, base.lon];

  return (
    <Box
      sx={{
        bgcolor: "background.paper",
        borderColor: "secondary.main",
        borderWidth: 2,
        borderStyle: "solid",
        borderRadius: 1,
        p: 1.5,
        height: { xs: 320, md: 520 },
        overflow: "hidden",
      }}
    >
      <MapContainer
        center={center}
        zoom={11}
        style={{ width: "100%", height: "100%", borderRadius: 12 }}
        scrollWheelZoom
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapFocusController />

        <TargetClickHandler
          enabled={isPickingTarget}
          onSelect={(lat, lon) => setTargetCoords({ lat, lon })}
        />

        <Marker position={[base.lat, base.lon]} icon={baseIcon}>
          <Popup>Basis</Popup>
        </Marker>

        <Marker position={[target.lat, target.lon]} icon={targetIcon}>
          <Popup>
            Ziel
            <br />
            {target.lat.toFixed(5)}, {target.lon.toFixed(5)}
          </Popup>
        </Marker>

        {robots.map((r) => {
          const isSelected = selectedRobotIds.includes(r.id);
          return (
            <Marker
              key={r.id}
              position={[r.position.lat, r.position.lon]}
              icon={isSelected ? robotSelectedIcon : robotIcon}
            >
              <Popup>
                {r.name} ({r.mode})
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </Box>
  );
}
