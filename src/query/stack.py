"""The project's real technology stack, read from its own build files (2026-10-15).

The same module as coding_agent's ``src/coding/stack.py``: both agents must
read a repository's versions the same way.

A design or a change must fit the versions the repository is built with: the
bank service targets Java 17, and a JDK-21-only call (``List.getFirst()``)
compiled there unnoticed, because ``maven.compiler.source/target`` do not check
the API — only ``--release`` does. So the stack is read here, deterministically,
from the build files, never guessed:

    pom.xml            Java version, Spring Boot parent, dependencies — and every
                       module's own pom.xml, with the parent's properties
    build.gradle(.kts) Java version (sourceCompatibility / toolchain), plugins —
                       and gradle/libs.versions.toml, the version catalog
    package.json       Node engine, dependencies
    pyproject.toml     requires-python, dependencies
    go.mod             Go version

A dependency with no version of its own under a Spring Boot parent gets the one
Spring Boot manages, from its published BOM (2026-10-15): "Spring Boot 3.2.5" says
nothing about Jackson, and the code must fit the Jackson that is really there.

``read_stack`` takes a repository's file list and a way to read one file, so a
caller reading through GitHub (Planning) and one with a checkout (Coding) read
it the same way.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache

import httpx

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


@dataclass(frozen=True)
class _Pom:
    stack: ProjectStack
    props: dict[str, str]
    modules: tuple[str, ...] = ()
    boot: str = ""  # the Spring Boot parent's version
    unversioned: tuple[str, ...] = ()  # "group:artifact" with no version of its own


def parse_pom(text: str) -> ProjectStack:
    return _read_pom(text).stack


def _read_pom(text: str, inherited: dict[str, str] | None = None) -> _Pom:
    try:
        root = _strip_ns(ET.fromstring(text))
    except ET.ParseError:
        return _Pom(ProjectStack(build_tool="Maven"), dict(inherited or {}))
    properties = root.find("properties")
    own = {child.tag: _text(child) for child in properties} if properties is not None else {}
    props = {**(inherited or {}), **own}

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
    boot = ""
    parent = root.find("parent")
    if parent is not None and _text(parent.find("groupId")) == "org.springframework.boot":
        boot = resolve(_text(parent.find("version")))
        frameworks.append(f"Spring Boot {boot}".strip())

    libraries: list[str] = []
    unversioned: list[str] = []
    for dep in root.findall("./dependencies/dependency"):
        name = f"{_text(dep.find('groupId'))}:{_text(dep.find('artifactId'))}"
        version = resolve(_text(dep.find("version")))
        libraries.append(f"{name} {version}".strip())
        if not version:
            unversioned.append(name)
    stack = ProjectStack(
        build_tool="Maven",
        language=f"Java {release}" if release else "",
        java_release=release,
        source=f"pom.xml ({where})" if where else "pom.xml",
        frameworks=tuple(frameworks),
        libraries=tuple(libraries[:MAX_LIBRARIES]),
    )
    modules = tuple(_text(m) for m in root.findall("./modules/module") if _text(m))
    return _Pom(stack, props, modules, boot, tuple(unversioned))


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


# --- the whole picture: modules, Spring Boot's managed versions, Gradle catalogs ------

SPRING_BOOT_BOM = (
    "https://repo1.maven.org/maven2/org/springframework/boot/spring-boot-dependencies/"
    "{v}/spring-boot-dependencies-{v}.pom"
)


@lru_cache(maxsize=16)
def spring_boot_managed(version: str) -> tuple[dict[str, str], dict[str, str]]:
    """({"group:artifact": version}, {bom group: its version}) Spring Boot ``version``
    manages, from its published BOM; ({}, {}) when it cannot be read."""
    if not version or os.environ.get("STACK_OFFLINE") == "1":
        return {}, {}
    try:
        resp = httpx.get(SPRING_BOOT_BOM.format(v=version), timeout=10)
        resp.raise_for_status()
        root = _strip_ns(ET.fromstring(resp.text))
    except (httpx.HTTPError, ET.ParseError):
        return {}, {}
    properties = root.find("properties")
    props = {c.tag: _text(c) for c in properties} if properties is not None else {}
    props.setdefault("project.version", version)

    def resolve(value: str) -> str:
        return re.sub(r"\$\{([^}]+)\}", lambda m: props.get(m.group(1), m.group(0)), value)

    managed: dict[str, str] = {}
    boms: dict[str, str] = {}
    for dep in root.findall("./dependencyManagement/dependencies/dependency"):
        group, artifact = _text(dep.find("groupId")), _text(dep.find("artifactId"))
        dep_version = resolve(_text(dep.find("version")))
        if _text(dep.find("scope")) == "import":
            boms.setdefault(group, dep_version)
        else:
            managed[f"{group}:{artifact}"] = dep_version
    return managed, boms


def _managed_version(name: str, boot: str) -> str:
    managed, boms = spring_boot_managed(boot)
    if name in managed:
        return managed[name]
    group = name.split(":", 1)[0]
    for bom_group, version in boms.items():
        if group == bom_group or group.startswith(bom_group + "."):
            return version
    return ""


def _with_boot_versions(found: ProjectStack, boot: str, unversioned: list[str]) -> ProjectStack:
    if not boot or not unversioned:
        return found
    named = []
    for library in found.libraries:
        if library in unversioned:
            version = _managed_version(library, boot)
            library = (
                f"{library} {version} (managed by Spring Boot {boot})"
                if version
                else f"{library} (version managed by Spring Boot {boot})"
            )
        named.append(library)
    return replace(found, libraries=tuple(named))


def _merged(first: ProjectStack, more: list[tuple[str, ProjectStack]]) -> ProjectStack:
    """A multi-module build as one stack: the Java version from the first module that
    sets it, every framework and library once."""
    release, language, source = first.java_release, first.language, first.source
    frameworks = list(first.frameworks)
    libraries = list(first.libraries)
    for path, module in more:
        if not release and module.java_release:
            release, language = module.java_release, module.language
            source = module.source.replace("pom.xml", path, 1)
        frameworks += [f for f in module.frameworks if f not in frameworks]
        libraries += [lib for lib in module.libraries if lib not in libraries]
    return replace(
        first,
        java_release=release,
        language=language,
        source=source,
        frameworks=tuple(frameworks),
        libraries=tuple(libraries[:MAX_LIBRARIES]),
    )


def _read_maven(paths: set[str], read: Callable[[str], str]) -> ProjectStack:
    root = _read_pom(read("pom.xml"))
    more: list[tuple[str, ProjectStack]] = []
    unversioned = list(root.unversioned)
    boot = root.boot
    pending = [(m, root.props) for m in root.modules]
    while pending and len(more) < 20:
        module, inherited = pending.pop(0)
        path = f"{module.rstrip('/')}/pom.xml"
        if path not in paths:
            continue
        child = _read_pom(read(path), inherited)
        more.append((path, child.stack))
        unversioned += [u for u in child.unversioned if u not in unversioned]
        boot = boot or child.boot
        pending += [(f"{module.rstrip('/')}/{m}", child.props) for m in child.modules]
    found = _merged(root.stack, more)
    return _with_boot_versions(found, boot, unversioned)


def parse_version_catalog(text: str) -> tuple[str, list[str], list[str]]:
    """(Java version, libraries, frameworks) from gradle/libs.versions.toml."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return "", [], []
    versions = {k: str(v) for k, v in (data.get("versions") or {}).items()}

    def version_of(entry: dict) -> str:
        v = entry.get("version")
        if isinstance(v, dict):
            return versions.get(str(v.get("ref", "")), "")
        if isinstance(entry.get("version.ref"), str):
            return versions.get(entry["version.ref"], "")
        return str(v or "")

    libraries = []
    for entry in (data.get("libraries") or {}).values():
        if isinstance(entry, str):
            group, artifact, *rest = entry.split(":") + [""]
            libraries.append(f"{group}:{artifact} {rest[0]}".strip())
            continue
        module = entry.get("module") or f"{entry.get('group', '')}:{entry.get('name', '')}"
        libraries.append(f"{module} {version_of(entry)}".strip())
    frameworks = []
    for entry in (data.get("plugins") or {}).values():
        if isinstance(entry, dict) and entry.get("id") == "org.springframework.boot":
            frameworks.append(f"Spring Boot {version_of(entry)}".strip())
    java = next((versions[k] for k in ("java", "jdk", "javaVersion") if k in versions), "")
    return java, libraries, frameworks


def _read_gradle(name: str, paths: set[str], read: Callable[[str], str]) -> ProjectStack:
    found = parse_gradle(read(name))
    found = replace(found, source=name)
    if "gradle/libs.versions.toml" not in paths:
        return found
    java, libraries, frameworks = parse_version_catalog(read("gradle/libs.versions.toml"))
    if java and not found.java_release:
        found = replace(
            found,
            java_release=java,
            language=f"Java {java}",
            source="gradle/libs.versions.toml (versions.java)",
        )
    return replace(
        found,
        frameworks=tuple(dict.fromkeys(found.frameworks + tuple(frameworks))),
        libraries=tuple(libraries[:MAX_LIBRARIES]),
    )


def read_stack(paths: list[str], read: Callable[[str], str]) -> ProjectStack:
    """The repository's stack from its file list and a way to read one file."""
    present = set(paths)
    for name, parse in PARSERS:
        if name not in present:
            continue
        if name == "pom.xml":
            return _read_maven(present, read)
        if name.startswith("build.gradle"):
            return _read_gradle(name, present, read)
        return parse(read(name))
    return ProjectStack()


_SKIP_DIRS = {".git", "node_modules", "target", "build", "dist", ".gradle", ".venv", "venv"}
