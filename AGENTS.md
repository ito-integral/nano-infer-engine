## Commit messages

Whenever the user asks for a commit message or a `git commit` command:

- Use Conventional Commits format for the subject.
- Always include the following trailer after a blank line:
  `Co-authored-by: Codex <codex@openai.com>`
- When providing a shell command, pass the trailer as a separate `-m` argument.
- Also recommend one concise, concrete next step based on the current project
  state after providing the commit message.
