export type Telemetry = {
  coords: string;
  speedKmh: number;
  battery: number;
  heading: string;
  altitude: number;
};

export const MOCK_TELEMETRY: Telemetry = {
  coords: `49°27'04.1"N 39°17'24.7"E`,
  speedKmh: 30,
  battery: 30,
  heading: "Північ",
  altitude: 172,
};
