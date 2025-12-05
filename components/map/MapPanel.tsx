"use client";

import { Box } from "@mui/material";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMapEvents,
  useMap,
} from "react-leaflet";
import { useTracking } from "../../context/TrackingContext";
import { useEffect, useState } from "react";
import { baseIcon, targetIcon, robotSelectedIcon, robotIcon } from "./icons";
import { MapStyleKey, TILE_LAYERS } from "./styles";
import { MapStyleControl } from "./MapStyleControl";

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
  const { mapFocus, robots, target, clearMapFocus } = useTracking();
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
    clearMapFocus();
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
  const [mapStyle, setMapStyle] = useState<MapStyleKey>("cartoLight");
  const center: [number, number] = [base.lat, base.lon];
  const layer = TILE_LAYERS[mapStyle];

  return (
    <Box
      sx={{
        bgcolor: "background.paper",
        borderColor: "secondary.main",
        borderWidth: 2,
        borderStyle: "solid",
        p: 1.5,
        height: { xs: 320, md: 520 },
        overflow: "hidden",
        position: "relative",
      }}
    >
      <MapContainer
        center={center}
        zoom={11}
        style={{ width: "100%", height: "100%" }}
        scrollWheelZoom
      >
        <TileLayer attribution={layer.attribution} url={layer.url} />

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
      <Box
        sx={{
          position: "absolute",
          top: 16,
          right: 16,
          zIndex: 1000,
        }}
      >
        <MapStyleControl value={mapStyle} onChange={setMapStyle} />
      </Box>
    </Box>
  );
}
