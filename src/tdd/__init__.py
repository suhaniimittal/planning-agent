"""The planner's TDD flow without stored code (2026-11-02).

Input: a Jira card, the services ranked for it, their pull requests and the
questions asked about it. Output: a plain-English TDD with no code in it, and
a work order for the coding agent. The code is seen only through GitHub's
REST API as paths, never cloned, never stored, and no file's contents are
read: the coding agent reads the code.
"""
