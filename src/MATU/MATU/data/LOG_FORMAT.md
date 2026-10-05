# Conversation Log Format

MATU starts from collected multi-agent conversation logs. The generation
framework can be CAMEL, AutoGen, a custom agent graph, or manual logs.

The expected JSON shape is:

```json
{
  "task_id_1": [
    [
      {"role": "user", "output": "first user turn"},
      {"role": "assistant", "output": "first assistant turn"}
    ],
    [
      {"role": "user", "output": "another run user turn"},
      {"role": "assistant", "output": "another run assistant turn"}
    ]
  ]
}
```

Conventions:

- Top-level keys are task/question ids.
- Each key maps to repeated runs for the same task.
- Each run is an ordered list of turns.
- Each turn needs a `role` field and either an `output` or `content` field.
- Roles are flexible. The default examples use `user` and `assistant`, but
  MATU can embed any roles passed to `matu/embed_logs.py --roles`.

