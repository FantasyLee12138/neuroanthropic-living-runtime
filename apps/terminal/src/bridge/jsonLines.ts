export function createJsonLineParser(onEvent: (payload: Record<string, unknown>) => void): (chunk: string) => void {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk;
    while (true) {
      const breakIndex = buffer.indexOf("\n");
      if (breakIndex < 0) {
        return;
      }
      const line = buffer.slice(0, breakIndex).trim();
      buffer = buffer.slice(breakIndex + 1);
      if (!line) {
        continue;
      }
      onEvent(JSON.parse(line) as Record<string, unknown>);
    }
  };
}
