import type { OutboundBridgeEvent } from "./types.js";

export function createOneShotPrinter(output: { write: (text: string) => void }): (event: OutboundBridgeEvent) => void {
  let streamedAssistant = false;
  let lineOpen = false;

  return (event: OutboundBridgeEvent): void => {
    if (event.type === "assistant_token") {
      if (!streamedAssistant) {
        output.write(`NALR: ${event.delta}`);
        streamedAssistant = true;
        lineOpen = true;
        return;
      }
      output.write(event.delta);
      return;
    }
    if (event.type === "assistant_final") {
      if (streamedAssistant) {
        if (lineOpen) {
          output.write("\n");
          lineOpen = false;
        }
        return;
      }
      output.write(`NALR: ${event.message}\n`);
      return;
    }
    if (event.type === "error") {
      if (lineOpen) {
        output.write("\n");
        lineOpen = false;
      }
      output.write(`Error: ${event.message}\n`);
    }
  };
}
