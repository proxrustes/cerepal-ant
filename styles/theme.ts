import { PaletteMode, createTheme } from "@mui/material";

export const getTheme = (mode: PaletteMode) =>
  createTheme({
    palette: {
      primary: {
        main: "#2A3C24",
      },
      secondary: {
        main: "#ffb300",
      },
    },
    shape: {
      borderRadius: 12,
    },
  });
