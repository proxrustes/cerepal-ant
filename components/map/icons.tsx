import L from "leaflet";
import { renderToString } from "react-dom/server";
import HomeIcon from "@mui/icons-material/Home";
import FlagIcon from "@mui/icons-material/Flag";
import PestControlIcon from "@mui/icons-material/PestControl";

export const robotIcon = L.divIcon({
  html: renderToString(
    <PestControlIcon style={{ fontSize: 42, color: "black" }} />
  ),
  className: "mui-leaflet-marker",
  iconSize: [30, 30],
  iconAnchor: [15, 30],
});

export const robotSelectedIcon = L.divIcon({
  html: renderToString(
    <PestControlIcon style={{ fontSize: 42, color: "orange" }} />
  ),
  className: "mui-leaflet-marker",
  iconSize: [30, 30],
  iconAnchor: [15, 30],
});

export const baseIcon = L.divIcon({
  html: renderToString(<HomeIcon style={{ fontSize: 28, color: "green" }} />),
  className: "mui-leaflet-marker",
  iconSize: [28, 28],
  iconAnchor: [14, 28],
});

export const targetIcon = L.divIcon({
  html: renderToString(<FlagIcon style={{ fontSize: 28, color: "red" }} />),
  className: "mui-leaflet-marker",
  iconSize: [28, 28],
  iconAnchor: [14, 28],
});
