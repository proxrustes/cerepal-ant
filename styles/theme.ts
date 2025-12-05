import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#86A5D9",
    },
    secondary: {
      main: "#344966",
    },
    background: {
      default: "#080B18",
      paper: "#161822",
    },
    divider: "rgba(148, 163, 184, 0.35)",
    text: {
      primary: "#e5e7eb",
      secondary: "#9ca3af",
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
