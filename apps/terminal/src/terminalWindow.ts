export function buildTerminalTitleSequence(title: string): string {
  return `\u001b]0;${title}\u0007`;
}

export function setTerminalTitle(
  title: string,
  options?: {
    isTTY?: boolean;
    write?: (text: string) => void;
    setProcessTitle?: (value: string) => void;
  },
): void {
  const isTTY = options?.isTTY ?? Boolean(process.stdout.isTTY);
  const write = options?.write ?? ((text: string) => process.stdout.write(text));
  const setProcessTitleFn = options?.setProcessTitle ?? ((value: string) => {
    process.title = value;
  });

  setProcessTitleFn(title);
  if (!isTTY) {
    return;
  }
  write(buildTerminalTitleSequence(title));
}
