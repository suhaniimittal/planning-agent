"""The planner's TDD flow without stored code (2026-11-02).

Input: a Jira card, the services ranked for it, their pull requests and the
questions asked about it. Output: a plain-English TDD with no code in it, and
a work order for the coding agent. The code is seen only through GitHub's
REST API, never cloned, never stored; ALLOW_CODE_READS decides whether a
file's text may be read into memory for the run at all (config.py).
"""
