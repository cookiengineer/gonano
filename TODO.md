
# TODO

- [ ] Model's tool protocol is expression-based, not OpenAI's JSON-schema {"name","arguments"} format
- [ ] The server therefore maps a tool call's raw expression into function.arguments and infers the tool name from the registry
- [ ] The model needs to emit name+JSON-schema calls that an arbitrary OpenAI client could execute against its own functions

