export function bearingDeg(
  from: { lat: number; lon: number },
  to: { lat: number; lon: number }
): number {
  const φ1 = (from.lat * Math.PI) / 180;
  const φ2 = (to.lat * Math.PI) / 180;
  const λ1 = (from.lon * Math.PI) / 180;
  const λ2 = (to.lon * Math.PI) / 180;
  const y = Math.sin(λ2 - λ1) * Math.cos(φ2);
  const x =
    Math.cos(φ1) * Math.sin(φ2) -
    Math.sin(φ1) * Math.cos(φ2) * Math.cos(λ2 - λ1);
  const θ = Math.atan2(y, x);
  return (θ * 180) / Math.PI;
}

export function toCardinalDE(deg: number): string {
  const d = (deg + 360) % 360;
  const dirs = ["N", "NO", "O", "SO", "S", "SW", "W", "NW"];
  const idx = Math.round(d / 45) % 8;
  return dirs[idx];
}
