// src/theme.ts
import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#86A5D9", // steel / blue
    },
    secondary: {
      main: "#344966",
    },
    background: {
      default: "#080B18", // slate-950
      paper: "#080B18", // чтобы всё было ровно одного тона
    },
    divider: "rgba(148, 163, 184, 0.35)", // slate-400
    text: {
      primary: "#e5e7eb", // slate-200
      secondary: "#9ca3af", // slate-400
    },
  },

  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          "&.Mui-selected": {
            backgroundColor: "rgba(148, 163, 184, 0.16)",
            "&:hover": {
              backgroundColor: "rgba(148, 163, 184, 0.24)",
            },
          },
        },
      },
    },
  },
});

export default theme;
