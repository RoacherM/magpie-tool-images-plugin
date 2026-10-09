import assert from 'node:assert/strict';
import test from 'node:test';
import { apply, moveToolImages } from './index.js';

const img = (id, name) => ({ type: 'image', attachment: { attachmentId: id, mediaType: 'image/png', bytes: 1, width: 1, height: 1, name } });
const text = (t) => ({ type: 'text', text: t });
const freeze = (value) => Object.freeze(value);

test('images leave the tool results and follow the batch in one user message', () => {
  const messages = freeze([
    { id: 'u1', role: 'user', content: [text('look')] },
    { id: 'a1', role: 'assistant', content: [{ type: 'tool-call', id: 'c1' }, { type: 'tool-call', id: 'c2' }, { type: 'tool-call', id: 'c3' }] },
    freeze({ id: 't1', role: 'tool', toolCallId: 'c1', content: freeze([text('<path>a.png</path>'), img('A', 'a.png')]) }),
    freeze({ id: 't2', role: 'tool', toolCallId: 'c2', content: freeze([text('no image here')]) }),
    freeze({ id: 't3', role: 'tool', toolCallId: 'c3', content: freeze([img('B'), text('after')]) }),
    { id: 'a2', role: 'assistant', content: [text('ok')] },
  ]);
  const out = moveToolImages(messages);
  assert.deepEqual(out.map((m) => `${m.id}:${m.role}`), ['u1:user', 'a1:assistant', 't1:tool', 't2:tool', 't3:tool', 't3:tool-images:user', 'a2:assistant']);
  assert.ok(out.slice(0, 5).every((m) => m.content.every((b) => b.type !== 'image')), 'no image left in tool results');
  assert.equal(out[2].content[1].text, '<tool_image index="1" name="a.png">shown in the next message\'s <tool_images></tool_image>');
  assert.equal(out[4].content[0].text, '<tool_image index="2">shown in the next message\'s <tool_images></tool_image>');
  assert.equal(out[4].toolCallId, 'c3', 'ids are kept unless renaming is asked for');
  assert.equal(out[1], messages[1]);
  assert.deepEqual(out[5].content.filter((b) => b.type === 'image').map((b) => b.attachment.attachmentId), ['A', 'B']);
  assert.match(out[5].content[0].text, /^<tool_images /);
  assert.equal(out[5].content.at(-1).text, '</tool_images>');
  assert.equal(messages[2].content[1].type, 'image', 'the original request is untouched');
});

test('every call of a batch with images gets a derived id Magpie cannot match; other batches keep theirs', () => {
  const messages = freeze([
    { id: 'u1', role: 'user', content: [text('go')] },
    { id: 'a0', role: 'assistant', content: [text('first'), { type: 'tool-call', id: 'plain' }] },
    { id: 't0', role: 'tool', toolCallId: 'plain', source: { kind: 'tool', callId: 'plain' }, content: [text('ok')] },
    { id: 'a1', role: 'assistant', content: [{ type: 'reasoning', text: 'r' }, { type: 'tool-call', id: 'shot', name: 'read_image' }, { type: 'tool-call', id: 'ls' }] },
    { id: 't1', role: 'tool', toolCallId: 'shot', source: { kind: 'tool', callId: 'shot' }, content: [img('A')] },
    { id: 't2', role: 'tool', toolCallId: 'ls', source: { kind: 'tool', callId: 'ls' }, content: [text('files')] },
  ]);
  const out = moveToolImages(messages, { rename: true });
  assert.equal(out[1], messages[1], 'a batch without images is passed through as is');
  assert.equal(out[2], messages[2]);
  assert.deepEqual(out[3].content.filter((b) => b.type === 'tool-call').map((b) => b.id), ['shot_img', 'ls_img']);
  assert.deepEqual(out[3].content[0], { type: 'reasoning', text: 'r' });
  assert.equal(out[3].content[1].name, 'read_image');
  assert.deepEqual([out[4].toolCallId, out[4].source.callId, out[5].toolCallId, out[5].source.callId], ['shot_img', 'shot_img', 'ls_img', 'ls_img']);
  assert.equal(out[6].role, 'user');
  assert.deepEqual(moveToolImages(messages, { rename: true }), out, 'the rewrite is deterministic, so later requests replay the same history');
  assert.equal(messages[3].content[1].id, 'shot', 'the original request is untouched');
});

test('derived ids stay within the Anthropic tool id limit', () => {
  const long = 'x'.repeat(64);
  const [, assistant, tool] = moveToolImages([
    { id: 'u', role: 'user', content: [text('go')] },
    { id: 'a', role: 'assistant', content: [{ type: 'tool-call', id: long }] },
    { id: 't', role: 'tool', toolCallId: long, content: [img('A')] },
  ], { rename: true });
  assert.match(assistant.content[0].id, /^[A-Za-z0-9_-]{1,64}$/);
  assert.equal(assistant.content[0].id, tool.toolCallId);
  assert.notEqual(tool.toolCallId, long);
});

test('requests without tool images are returned as the same array', () => {
  const messages = [{ id: 'u', role: 'user', content: [img('X')] }, { id: 't', role: 'tool', content: [text('x')] }];
  assert.equal(moveToolImages(messages), messages);
});

function mount(config) {
  let listener;
  const dispatched = [];
  const ctx = {
    on: (event, fn, options) => { assert.equal(event, 'llm/stream'); assert.equal(options.global, true); listener = fn; },
    // ctx.llm.stream re-enters the waterfall: simulate by calling the listener again.
    llm: { stream: (options) => listener(options, () => { dispatched.push(options); return 'stream'; }) },
  };
  apply(ctx, config);
  return { listen: (...args) => listener(...args), dispatched };
}

const imageRequest = (model, provider = 'magpie') => freeze({
  provider,
  model,
  messages: freeze([
    { id: 'a', role: 'assistant', content: [{ type: 'tool-call', id: 'c1' }] },
    { id: 't', role: 'tool', toolCallId: 'c1', content: [img('A')] },
  ]),
});

test('codex routes get images moved out, once, with their ids kept', () => {
  const { listen, dispatched } = mount();
  assert.equal(listen(imageRequest('codex/gpt-6-sol'), () => assert.fail('the original call must not continue')), 'stream');
  assert.equal(dispatched.length, 1);
  assert.equal(dispatched[0].model, 'codex/gpt-6-sol');
  assert.equal(dispatched[0].messages[1].toolCallId, 'c1');
  assert.equal(dispatched[0].messages[2].content.find((b) => b.type === 'image').attachment.attachmentId, 'A');
});

test('claude routes and other providers pass through unless configured', () => {
  const { listen, dispatched } = mount();
  for (const request of [imageRequest('claude/claude-opus-5-5'), imageRequest('codex/gpt-6-sol', 'deepseek')]) {
    let passed = false;
    listen(request, () => { passed = true; });
    assert.ok(passed, `${request.provider} ${request.model} is left alone`);
  }
  assert.equal(dispatched.length, 0);
});

test('renameModels opts a route into derived ids', () => {
  const { listen, dispatched } = mount({ renameModels: ['claude/'] });
  assert.equal(listen(imageRequest('claude/claude-opus-5-5'), () => assert.fail('the original call must not continue')), 'stream');
  assert.equal(dispatched[0].messages[0].content[0].id, 'c1_img');
  assert.equal(dispatched[0].messages[1].toolCallId, 'c1_img');
});
