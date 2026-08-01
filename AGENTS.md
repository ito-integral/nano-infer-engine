## Commit messages

Whenever the user asks for a commit message or a `git commit` command:

- Use Conventional Commits format for the subject.
- Always include the following trailer after a blank line:
  `Co-authored-by: Codex <codex@openai.com>`
- When providing a shell command, pass the trailer as a separate `-m` argument.
- Also recommend one concise, concrete next step based on the current project
  state after providing the commit message.

## Roadmap maintenance

Whenever the user asks for an implementation or behavior change:

- Review and update `docs/development-roadmap.md` in the same task.
- Keep the roadmap consistent with what was actually implemented, tested,
  deferred, or left as a known limitation.
- Write roadmap updates in Chinese.
- Do not update the roadmap for explanation-only, review-only, or commit-message-only
  requests unless the project state changed.
