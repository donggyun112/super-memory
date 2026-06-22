// correct() reuses the prior keys when the caller passes none. If the correction
// changed the topic, those keys silently mis-tag the new memory. buildRetagNote
// returns a warning naming the reused keys in that case, and null otherwise.
export function buildRetagNote(
  callerKeys: unknown,
  retainedKeys: string[]
): string | null {
  const callerProvidedKeys = Array.isArray(callerKeys) && callerKeys.length > 0;
  if (callerProvidedKeys) return null;
  if (retainedKeys.length === 0) return null;
  return `Reused ${retainedKeys.length} prior key(s) [${retainedKeys.join(", ")}] — pass \`keys\` to retag if the correction changed the topic.`;
}
