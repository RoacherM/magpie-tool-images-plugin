/**
 * The Magpie proxy (127.0.0.1:3425) drops image blocks inside Anthropic `tool_result`s but keeps
 * images in user messages. For requests routed to the configured providers, this moves each image
 * out of a batch of tool results into one user message right after it, leaving a text pointer in
 * its place. Only the outgoing request changes; the session history keeps the original results.
 *
 * What each Magpie route needs (measured 2026-09-28 against Magpie directly):
 * - `codex/`: moving is enough, and the prompt cache stays warm.
 * - `claude/`: Magpie drives a live Claude Code session; a request that answers the call that
 *   session is waiting on hands back the tool_result text only and drops what follows. Giving
 *   every call of the image batch a derived id makes Magpie rebuild the conversation, which keeps
 *   the image, but every later step then misses the prompt cache. So that route is left alone
 *   unless its model prefix is listed in `renameModels`.
 * Leaving `claude/` untouched also avoids a pointer to an image that never arrives.
 *
 * The agent loop freezes its request, so the listener cannot edit it in place: it starts the
 * rewritten call through `ctx.llm.stream` instead, which passes through every `llm/stream`
 * listener once (this one then finds nothing to move and continues).
 */
export const name = 'dsh-magpie-tool-images';
export const inject = ['llm'];

/** Anthropic tool ids allow `[A-Za-z0-9_-]{1,64}`. */
export const imageCallId = (id) => `${String(id).slice(0, 60)}_img`;

/** Ids of every tool call issued alongside a call whose result carries an image. */
function imageBatchCalls(messages) {
  const withImages = new Set();
  for (const message of messages) {
    if (message.role === 'tool' && message.content.some((block) => block.type === 'image')) withImages.add(message.toolCallId);
  }
  const ids = new Set();
  if (withImages.size === 0) return ids;
  for (const message of messages) {
    if (message.role !== 'assistant') continue;
    const calls = message.content.filter((block) => block.type === 'tool-call');
    if (calls.some((block) => withImages.has(block.id))) for (const block of calls) ids.add(block.id);
  }
  return ids;
}

/**
 * @param options.rename - also give image batches derived tool-call ids.
 * @returns the rewritten message list, or the same array when there is nothing to move.
 */
export function moveToolImages(messages, { rename = false } = {}) {
  const renamed = rename ? imageBatchCalls(messages) : new Set();
  let changed = false;
  const out = [];
  let batch = [];
  let count = 0;
  const flush = () => {
    if (batch.length === 0) return;
    out.push({
      id: out[out.length - 1].id + ':tool-images',
      role: 'user',
      source: { kind: 'user' },
      content: [
        { type: 'text', text: `<tool_images note="Images returned by the tool call${batch.length > 1 ? 's' : ''} above, in order. Not written by the user.">` },
        ...batch.flatMap((block, index) => [{ type: 'text', text: `<tool_image index="${index + 1}"/>` }, block]),
        { type: 'text', text: '</tool_images>' },
      ],
    });
    batch = [];
  };
  for (const original of messages) {
    const message = renameCalls(original, renamed);
    if (message !== original) changed = true;
    if (message.role !== 'tool') flush();
    if (message.role === 'tool' && message.content.some((block) => block.type === 'image')) {
      changed = true;
      const content = message.content.map((block) => {
        if (block.type !== 'image') return block;
        batch.push(block);
        count += 1;
        const label = block.attachment?.name ?? '';
        return { type: 'text', text: `<tool_image index="${count}"${label ? ` name="${block.attachment.name.replace(/"/g, '')}"` : ''}>shown in the next message's <tool_images></tool_image>` };
      });
      out.push({ ...message, content });
      continue;
    }
    out.push(message);
  }
  flush();
  return changed ? out : messages;
}

/** Give the calls in `ids` their derived id on both the assistant tool-call and its result. */
function renameCalls(message, ids) {
  if (message.role === 'assistant' && message.content.some((block) => block.type === 'tool-call' && ids.has(block.id))) {
    return { ...message, content: message.content.map((block) => (block.type === 'tool-call' && ids.has(block.id) ? { ...block, id: imageCallId(block.id) } : block)) };
  }
  if (message.role === 'tool' && ids.has(message.toolCallId)) {
    const toolCallId = imageCallId(message.toolCallId);
    return { ...message, toolCallId, ...(message.source?.callId === undefined ? {} : { source: { ...message.source, callId: toolCallId } }) };
  }
  return message;
}

const matches = (prefixes, model) => prefixes.some((prefix) => String(model ?? '').startsWith(prefix));

/**
 * @param config.providers - DSH provider ids routed through Magpie (default `['magpie']`).
 * @param config.moveModels - model prefixes that get images moved out (default `['codex/']`).
 * @param config.renameModels - model prefixes that also get derived ids; costs the prompt cache (default none).
 */
export function apply(ctx, config = {}) {
  const providers = new Set(config.providers ?? ['magpie']);
  const moveModels = config.moveModels ?? ['codex/'];
  const renameModels = config.renameModels ?? [];
  ctx.on('llm/stream', (options, next) => {
    if (!providers.has(options.provider)) return next();
    const rename = matches(renameModels, options.model);
    if (!rename && !matches(moveModels, options.model)) return next();
    const messages = moveToolImages(options.messages, { rename });
    if (messages === options.messages) return next();
    return ctx.llm.stream({ ...options, messages });
  }, { global: true });
}
