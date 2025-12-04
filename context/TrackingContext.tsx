// src/tracking/TrackingContext.tsx
"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  ReactNode,
} from "react";

export type LatLng = { lat: number; lon: number };
export type RobotMode = "idle" | "toBase" | "toTarget";

export type Robot = {
  id: string;
  name: string;
  position: LatLng;
  mode: RobotMode;
  battery: number;
};

type SelectedRobots = "all" | string;

type TrackingContextValue = {
  robots: Robot[];
  base: LatLng;
  target: LatLng;
  selected: SelectedRobots;
  setSelected: (sel: SelectedRobots) => void;
  commandToBase: () => void;
  commandToTarget: () => void;
  isPickingTarget: boolean;
  beginTargetSelection: () => void;
  setTargetCoords: (coords: LatLng) => void;
};

const TrackingContext = createContext<TrackingContextValue | undefined>(
  undefined
);

export const useTracking = () => {
  const ctx = useContext(TrackingContext);
  if (!ctx) throw new Error("useTracking must be used inside TrackingProvider");
  return ctx;
};

const BASE_CENTER: LatLng = { lat: 49.451139, lon: 39.290194 };
const TARGET_POINT: LatLng = { lat: 49.47, lon: 39.33 };

function randomAround(center: LatLng, delta = 0.01): LatLng {
  const rnd = () => (Math.random() * 2 - 1) * delta;
  return { lat: center.lat + rnd(), lon: center.lon + rnd() };
}

export function TrackingProvider({ children }: { children: ReactNode }) {
  const [base] = useState<LatLng>(BASE_CENTER);
  const [target, setTarget] = useState<LatLng>(TARGET_POINT);
  const [isPickingTarget, setIsPickingTarget] = useState(false);

  const [robots, setRobots] = useState<Robot[]>(() => [
    {
      id: "r1",
      name: "Ant-1",
      position: randomAround(BASE_CENTER, 0.05),
      mode: "idle",
      battery: 60 + Math.round(Math.random() * 40),
    },
    {
      id: "r2",
      name: "Ant-2",
      position: randomAround(BASE_CENTER, 0.01),
      mode: "idle",
      battery: 60 + Math.round(Math.random() * 40),
    },
  ]);

  const [selected, setSelected] = useState<SelectedRobots>("all");

  useEffect(() => {
    const STEP = 0.0005;

    const id = window.setInterval(() => {
      setRobots((prev) =>
        prev.map((r) => {
          const dest =
            r.mode === "toBase" ? base : r.mode === "toTarget" ? target : null;

          if (!dest) return r;

          const dLat = dest.lat - r.position.lat;
          const dLon = dest.lon - r.position.lon;
          const dist = Math.sqrt(dLat * dLat + dLon * dLon);

          const drain = 0.1; // % за тик
          const battery = Math.max(0, r.battery - drain);

          if (dist < STEP) {
            return { ...r, position: dest, mode: "idle", battery };
          }

          return {
            ...r,
            position: {
              lat: r.position.lat + (dLat / dist) * STEP,
              lon: r.position.lon + (dLon / dist) * STEP,
            },
            battery,
          };
        })
      );
    }, 200);

    return () => window.clearInterval(id);
  }, [base, target]);

  const commandToBase = () => {
    setRobots((prev) =>
      prev.map((r) =>
        selected === "all" || r.id === selected ? { ...r, mode: "toBase" } : r
      )
    );
  };

  const commandToTarget = () => {
    setRobots((prev) =>
      prev.map((r) =>
        selected === "all" || r.id === selected ? { ...r, mode: "toTarget" } : r
      )
    );
  };

  const beginTargetSelection = () => {
    setIsPickingTarget(true);
  };

  const setTargetCoords = (coords: LatLng) => {
    setTarget(coords);
    setIsPickingTarget(false);
  };

  const value = useMemo(
    () => ({
      robots,
      base,
      target,
      selected,
      setSelected,
      commandToBase,
      commandToTarget,
      isPickingTarget,
      beginTargetSelection,
      setTargetCoords,
    }),
    [robots, base, target, selected]
  );

  return (
    <TrackingContext.Provider value={value}>
      {children}
    </TrackingContext.Provider>
  );
}
