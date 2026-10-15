"""The project's real technology stack, read from its own build files (2026-10-15).

The same module as coding_agent's ``src/coding/stack.py``: both agents must
read a repository's versions the same way.

A design or a change must fit the versions the repository is built with: the
bank service targets Java 17, and a JDK-21-only call (``List.getFirst()``)
compiled there unnoticed, because ``maven.compiler.source/target`` do not check
the API — only ``--release`` does. So the stack is read here, deterministically,
from the build files, never guessed:

    pom.xml            Java version, Spring Boot parent, dependencies
    build.gradle(.kts) Java version (sourceCompatibility / toolchain), plugins
    package.json       Node engine, dependencies
    pyproject.toml     requires-python, dependencies
    go.mod             Go version

``parse_*`` take the file's text, so a caller that fetched the file (Planning,
through GitHub) and one that has a checkout (Coding) read it the same way.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# How many libraries to name: enough to show the stack, few enough to read.
MAX_LIBRARIES = 15


@dataclass(frozen=True)
class ProjectStack:
    build_tool: str = ""  # "Maven", "Gradle", "npm", "Python", "Go"
    language: str = ""  # "Java 17", "Node >=18", "Python >=3.12", "Go 1.22"
    java_release: str = ""  # "17" — what a Java build must be checked against
    source: str = ""  # where it was read: "pom.xml (maven.compiler.source/target)"
    frameworks: tuple[str, ...] = ()  # "Spring Boot 3.2.5"
    # "org.junit.jupiter:junit-jupiter 5.10.2"
    libraries: tuple[str, ...] = field(default_factory=tuple)

    def known(self) -> bool:
        return bool(self.build_tool or self.language)

    def constraints(self) -> list[str]:
        """The rules a design and a change must keep, in plain sentences."""
        if not self.known():
            return []
        out: list[str] = []
        if self.language:
            out.append(
                f"Language: {self.language} (from {self.source}). Use only APIs and syntax "
                f"available in {self.language}; nothing that needs a later version."
            )
        if self.build_tool:
            out.append(
                f"Build: {self.build_tool}. Do not add, remove or upgrade dependencies, "
                "plugins or versions; if the change truly needs one, say so as an open "
                "question instead of changing it."
            )
        if self.frameworks:
            out.append("Frameworks: " + ", ".join(self.frameworks) + ".")
        if self.libraries:
            out.append("Libraries in use: " + "; ".join(self.libraries) + ".")
        return out


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def parse_pom(text: str) -> ProjectStack:
    try:
        root = _strip_ns(ET.fromstring(text))
    except ET.ParseError:
        return ProjectStack(build_tool="Maven")
    properties = root.find("properties")
    props = {child.tag: _text(child) for child in properties} if properties is not None else {}

    def resolve(value: str) -> str:
        return re.sub(r"\$\{([^}]+)\}", lambda m: props.get(m.group(1), m.group(0)), value)

    release, where = "", ""
    for key in ("maven.compiler.release", "java.version", "maven.compiler.source"):
        if props.get(key):
            release, where = resolve(props[key]), key
            break
    if where == "maven.compiler.source":
        where = "maven.compiler.source/target"
    release = release.removeprefix("1.") if release.startswith("1.") else release

    frameworks: list[str] = []
    parent = root.find("parent")
    if parent is not None and _text(parent.find("groupId")) == "org.springframework.boot":
        frameworks.append(f"Spring Boot {resolve(_text(parent.find('version')))}".strip())

    libraries: list[str] = []
    for dep in root.findall("./dependencies/dependency"):
        name = f"{_text(dep.find('groupId'))}:{_text(dep.find('artifactId'))}"
        version = resolve(_text(dep.find("version")))
        libraries.append(f"{name} {version}".strip())
    return ProjectStack(
        build_tool="Maven",
        language=f"Java {release}" if release else "",
        java_release=release,
        source=f"pom.xml ({where})" if where else "pom.xml",
        frameworks=tuple(frameworks),
        libraries=tuple(libraries[:MAX_LIBRARIES]),
    )


_GRADLE_JAVA = (
    re.compile(r"languageVersion\s*(?:=|\.set\()\s*JavaLanguageVersion\.of\(\s*(\d+)\s*\)"),
    re.compile(
        r"(?:sourceCompatibility|targetCompatibility)\s*=\s*JavaVersion\.VERSION_(?:1_)?(\d+)"
    ),
    re.compile(r"(?:sourceCompatibility|targetCompatibility)\s*=\s*['\"]?(?:1\.)?(\d+)['\"]?"),
)


def parse_gradle(text: str) -> ProjectStack:
    release = ""
    for pattern in _GRADLE_JAVA:
        found = pattern.search(text)
        if found:
            release = found.group(1)
            break
    frameworks = []
    boot = re.search(
        r"id\s*\(?\s*['\"]org\.springframework\.boot['\"]\s*\)?\s*version\s*['\"]([^'\"]+)",
        text,
    )
    if boot:
        frameworks.append(f"Spring Boot {boot.group(1)}")
    return ProjectStack(
        build_tool="Gradle",
        language=f"Java {release}" if release else "",
        java_release=release,
        source="build.gradle",
        frameworks=tuple(frameworks),
    )


def parse_package_json(text: str) -> ProjectStack:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ProjectStack(build_tool="npm")
    node = str((data.get("engines") or {}).get("node") or "")
    deps = {**(data.get("dependencies") or {}), **(data.get("devDependencies") or {})}
    return ProjectStack(
        build_tool="npm",
        language=f"Node {node}" if node else "",
        source="package.json (engines.node)" if node else "package.json",
        libraries=tuple(f"{k} {v}" for k, v in list(deps.items())[:MAX_LIBRARIES]),
    )


def parse_pyproject(text: str) -> ProjectStack:
    python = re.search(r"requires-python\s*=\s*['\"]([^'\"]+)['\"]", text)
    block = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)\]", text)
    deps = re.findall(r"['\"]([^'\"]+)['\"]", block.group(1)) if block else []
    return ProjectStack(
        build_tool="Python",
        language=f"Python {python.group(1)}" if python else "",
        source="pyproject.toml (requires-python)" if python else "pyproject.toml",
        libraries=tuple(deps[:MAX_LIBRARIES]),
    )


def parse_go_mod(text: str) -> ProjectStack:
    go = re.search(r"(?m)^go\s+(\S+)", text)
    return ProjectStack(
        build_tool="Go",
        language=f"Go {go.group(1)}" if go else "",
        source="go.mod",
    )


# The build file each parser reads, in the order they are tried.
PARSERS = (
    ("pom.xml", parse_pom),
    ("build.gradle.kts", parse_gradle),
    ("build.gradle", parse_gradle),
    ("package.json", parse_package_json),
    ("pyproject.toml", parse_pyproject),
    ("go.mod", parse_go_mod),
)
BUILD_FILES = tuple(name for name, _ in PARSERS)


def from_files(files: dict[str, str]) -> ProjectStack:
    """The stack from build files already read (name -> text); the first found wins."""
    for name, parse in PARSERS:
        if files.get(name):
            return parse(files[name])
    return ProjectStack()


def detect(repo_root: Path) -> ProjectStack:
    """The stack of a checked-out repository, from the build file at its root."""
    for name, parse in PARSERS:
        path = repo_root / name
        if path.is_file():
            return parse(path.read_text(errors="replace"))
    return ProjectStack()
