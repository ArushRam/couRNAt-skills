# comp-bio-skills

This repo holds Claude Code skills for common computational biology work, for example Slurm jobs on the school cluster. A skill is a folder with a `SKILL.md` file. The file tells Claude how and when to do one task.

## Layout

```
skills/       One folder per skill. Each folder has a SKILL.md file.
template/     A starting SKILL.md to copy for a new skill.
```

A skill folder can also hold scripts, reference files, or example data that `SKILL.md` points to.

## Install the skills

Claude Code reads personal skills from `~/.claude/skills/`. To install all skills in this repo, run these commands from the repo root:

```bash
mkdir -p ~/.claude/skills
for d in skills/*/; do ln -sfn "$PWD/$d" ~/.claude/skills/"$(basename "$d")"; done
```

The links point to this repo, so `git pull` updates your skills. To use the skills in one project only, link them into `<project>/.claude/skills/` instead.

## Add a skill

1. Copy the template: `cp -r template skills/<skill-name>`.
2. Set `name` in the frontmatter to `<skill-name>`. Use lowercase letters, numbers, and hyphens only.
3. Write the `description`. Claude reads it to decide when to use the skill, so name the tasks and words that trigger it.
4. Write the steps. Keep `SKILL.md` short, and put long reference material in separate files in the same folder.
5. Run the install loop again, then test the skill in Claude Code.
6. Open a pull request.

## Rules for shared skills

Do not commit passwords, tokens, or private keys. Do not commit patient data or other protected data. If a skill needs a username or an account name, tell the user to supply it instead of writing your own into the file.
