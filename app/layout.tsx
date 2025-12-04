import type { Metadata } from "next";
import { ReactNode } from "react";
import ThemeRegistry from "../styles/ThemeRegistry";
import "leaflet/dist/leaflet.css";
import { TrackingProvider } from "../context/TrackingContext";

export const metadata: Metadata = {
  title: "Cerepal Ant",
  description: "Cerepal Ant dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ThemeRegistry>
          <TrackingProvider>{children}</TrackingProvider>
        </ThemeRegistry>
      </body>
    </html>
  );
}
