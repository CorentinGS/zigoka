const theme = {
  background: "#151413",
  panel: "#211f1c",
  border: "#3a3732",
  text: "#ece9e3",
  muted: "#a29d94",
  board: "#c49a48",
  line: "#3b2f1a",
  zig: "#3987e5",
  moka: "#d95926",
  draw: "#8a867e",
  font: "Inter, 'Noto Sans', system-ui, sans-serif",
};

const ENGINE_COLORS = { zig: theme.zig, moka: theme.moka } as const;

const PLAYER_NAMES = {
  "zig-search": "Zig + search in WASM",
  "zig-puct": "Zig + search",
  "moka-puct": "Moka + search",
  "zig-policy": "Zig, top move",
  "moka-policy": "Moka, top move",
} as const;

const getEngineColor = (kind: keyof typeof PLAYER_NAMES) =>
  ENGINE_COLORS[kind.startsWith("zig") ? "zig" : "moka"];

export { getEngineColor, PLAYER_NAMES, theme };
