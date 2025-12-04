"use client";

import { Box, Stack } from "@mui/material";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import HomeIcon from "@mui/icons-material/Home";
import FlagIcon from "@mui/icons-material/Flag";
import { renderToString } from "react-dom/server";
import { useTracking } from "../../context/TrackingContext";
import PestControlIcon from "@mui/icons-material/PestControl";

const robotIcon = L.divIcon({
  html: renderToString(
    <PestControlIcon style={{ fontSize: 42, color: "#2A3C24" }} />
  ),
  className: "mui-leaflet-marker",
  iconSize: [30, 30],
  iconAnchor: [15, 30],
});

const robotSelectedIcon = L.divIcon({
  html: renderToString(
    <PestControlIcon style={{ fontSize: 42, color: "#339989" }} />
  ),
  className: "mui-leaflet-marker",
  iconSize: [30, 30],
  iconAnchor: [15, 30],
});

const baseIcon = L.divIcon({
  html: renderToString(<HomeIcon style={{ fontSize: 28 }} />),
  className: "mui-leaflet-marker",
  iconSize: [28, 28],
  iconAnchor: [14, 28],
});

const targetIcon = L.divIcon({
  html: renderToString(<FlagIcon style={{ fontSize: 28 }} />),
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
export function MapPanel() {
  const { robots, base, target, selected, isPickingTarget, setTargetCoords } =
    useTracking();
  const center: [number, number] = [base.lat, base.lon];

  return (
    <Stack>
      <Box
        sx={{
          bgcolor: "background.paper",
          borderColor: "primary.main",
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
          <TargetClickHandler
            enabled={isPickingTarget}
            onSelect={(lat, lon) => setTargetCoords({ lat, lon })}
          />
          {/* база */}
          <Marker position={[base.lat, base.lon]} icon={baseIcon}>
            <Popup>База</Popup>
          </Marker>

          {/* цель */}
          <Marker position={[target.lat, target.lon]} icon={targetIcon}>
            <Popup>Ціль</Popup>
          </Marker>

          {/* роботы */}
          {robots.map((r) => {
            const isSelected = selected === "all" || selected === r.id;
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
    </Stack>
  );
}
