export type MapStyleKey =
  | "osmStandard"
  | "osmHumanitarian"
  | "cartoLight"
  | "cartoDark"
  | "esriImagery"
  | "esriStreets"
  | "esriTopographic";

export const TILE_LAYERS: Record<
  MapStyleKey,
  { url: string; attribution: string; label: string }
> = {
  osmStandard: {
    label: "OSM Standard",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors",
  },

  osmHumanitarian: {
    label: "OSM Humanitarian",
    url: "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
    attribution:
      '&copy; OpenStreetMap contributors, Tiles style by <a href="https://www.hotosm.org/">Humanitarian OpenStreetMap Team</a>',
  },

  cartoLight: {
    label: "Carto Light",
    url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attribution:
      '&copy; OpenStreetMap contributors &copy; <a href="https://carto.com/">CARTO</a>',
  },

  cartoDark: {
    label: "Carto Dark",
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution:
      '&copy; OpenStreetMap contributors &copy; <a href="https://carto.com/">CARTO</a>',
  },

  esriImagery: {
    label: "Esri Imagery",
    url:
      "https://server.arcgisonline.com/ArcGIS/rest/services/" +
      "World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution:
      "Tiles &copy; Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, " +
      "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
  },

  esriStreets: {
    label: "Esri Streets",
    url:
      "https://server.arcgisonline.com/ArcGIS/rest/services/" +
      "World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attribution:
      "Tiles &copy; Esri — Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, " +
      "iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012",
  },

  esriTopographic: {
    label: "Esri Topographic",
    url:
      "https://server.arcgisonline.com/ArcGIS/rest/services/" +
      "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    attribution:
      "Tiles &copy; Esri — Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, " +
      "Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012",
  },
};
