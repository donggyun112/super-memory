// correct() without explicit keys silently reuses the prior keys. If the correction
// changed the topic, the new memory is left mis-tagged with no signal. buildRetagNote
// surfaces a warning in that case, and stays silent when the caller passed keys.
import assert from "node:assert/strict";
import test from "node:test";
import { buildRetagNote } from "../src/retag.ts";

test("warns and lists reused keys when correct() got no explicit keys", () => {
  const note = buildRetagNote(undefined, ["strawberry", "딸기"]);
  assert.ok(note, "expected a note when keys were reused");
  assert.ok(note!.includes("strawberry") && note!.includes("딸기"), "note lists the reused keys");
  assert.match(note!, /retag|keys/i);
});

test("stays silent when the caller provided explicit keys", () => {
  assert.equal(buildRetagNote(["blueberry"], ["strawberry"]), null);
});

test("stays silent when there are no reused keys", () => {
  assert.equal(buildRetagNote(undefined, []), null);
});
