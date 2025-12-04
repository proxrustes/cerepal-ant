import type { Metadata } from "next";
import { ReactNode } from "react";
import ThemeRegistry from "../styles/ThemeRegistry";

export const metadata: Metadata = {
  title: "Cerepal Ant",
  description: "Cerepal Ant dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ThemeRegistry>{children}</ThemeRegistry>
      </body>
    </html>
  );
}
