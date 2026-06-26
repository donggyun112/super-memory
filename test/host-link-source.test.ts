// The remember/correct/remember_batch handlers stamp the active host-agent
// transcript link (host_session/host_agent/host_turn) onto a memory's source
// via buildSource, so a recalled memory can be traced back to its original
// conversation. Caller-provided source fields must still win.
import assert from "node:assert/strict";
import test from "node:test";

const { buildSource } = await import("../src/server.ts");

const HOST = { agent: "claude" as const, session_id: "e7f5b1d2-1602-4180-ac66-9f9acbd1f673", turn: 7 };

test("buildSource stamps the host transcript link when a session is active", () => {
  const source = buildSource(null, "remember", HOST);
  assert.equal(source.host_agent, "claude");
  assert.equal(source.host_session, "e7f5b1d2-1602-4180-ac66-9f9acbd1f673");
  assert.equal(source.host_turn, 7);
  assert.equal(source.tool, "remember");
});

test("buildSource omits host fields when no session is active", () => {
  const source = buildSource(null, "remember", null);
  assert.ok(!("host_session" in source));
  assert.ok(!("host_agent" in source));
});

test("caller-provided source overrides the auto-detected host link", () => {
  const source = buildSource(
    { host_session: "caller-supplied", conversation: "conv-9" },
    "remember",
    HOST
  );
  assert.equal(source.host_session, "caller-supplied"); // caller wins
  assert.equal(source.conversation, "conv-9");
  assert.equal(source.host_agent, "claude"); // un-overridden host field stays
});
