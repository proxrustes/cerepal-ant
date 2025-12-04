export type Telemetry = {
  coordsText: string;
  lat: number;
  lon: number;
  speedKmh: number;
  battery: number;
  heading: string;
  altitude: number;
};

export const MOCK_TELEMETRY: Telemetry = {
  coordsText: `49°27'04.1"N 39°17'24.7"E`,
  lat: 49.451139,
  lon: 39.290194,
  speedKmh: 30,
  battery: 30,
  heading: "NORD-OST",
  altitude: 172,
};
