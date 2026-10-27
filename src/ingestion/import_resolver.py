"""Turns import statements into the files they actually load.

The planner missed `InfiniteScrollGrid/view.tsx` on a real ticket because
nothing linked it to the screen that renders it: the screen imports
`@client/components/InfiniteScrollGrid`, which is an alias (tsconfig maps
`@client/*` to `src/apps/client/*`) for a folder whose `index.ts` holds
only `export { InfiniteScrollGrid } from "./view"`. Resolving the import to
the barrel would still not reach the component, so barrels are followed to
the file that DEFINES each imported name.

Never guesses: an import resolves only to a file that exists. An import
that looks local (relative, or matching an alias) but resolves to nothing
is returned as unresolved and reported; a package import (`react`,
`org.springframework...`) is simply external.
"""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .code_parser import ImportSpec

_JS_LANGUAGES = {"typescript", "tsx", "javascript"}
_JVM_LANGUAGES = {"kotlin", "java"}
_JS_EXTENSIONS = (
    ".ts",
    ".tsx",
    ".d.ts",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".graphql",
    ".gql",
)
_INDEX_FILES = ("index.ts", "index.tsx", "index.js", "index.jsx")
_TSCONFIG_NAME = re.compile(r"^(tsconfig(\.[\w-]+)?|jsconfig)\.json$")
# How many barrels deep a name is followed before giving up (a cycle of
# barrels re-exporting each other is caught separately).
_MAX_BARREL_DEPTH = 6


@dataclass
class ResolvedImport:
    source: str
    target: str
    names: list[str] = field(default_factory=list)  # local names this edge brings in
    via: list[str] = field(default_factory=list)  # barrels passed through
    reexport: bool = False


@dataclass
class FileImports:
    """Everything an import resolution needs to know about one file."""

    path: str
    language: str
    imports: list[ImportSpec]
    package: str | None = None
    top_level: set[str] = field(default_factory=set)  # names it declares at top level


@dataclass
class _TsConfig:
    directory: str
    base_url: str | None
    paths: dict[str, list[str]]


def _strip_jsonc(text: str) -> str:
    """JSON with comments and trailing commas (what tsconfig.json allows)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            end = i + 1
            while end < n and text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            out.append(text[i : end + 1])
            i = end + 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(ch)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _load_tsconfigs(sources: dict[str, str]) -> list[_TsConfig]:
    raw: dict[str, dict] = {}
    for path, text in sources.items():
        if _TSCONFIG_NAME.match(PurePosixPath(path).name):
            try:
                raw[path] = json.loads(_strip_jsonc(text))
            except (json.JSONDecodeError, ValueError):
                continue

    def options(path: str, depth: int = 0) -> dict:
        data = raw.get(path, {})
        merged: dict = {}
        parent = data.get("extends")
        if isinstance(parent, str) and depth < 5:
            parent_path = posixpath.normpath(posixpath.join(posixpath.dirname(path), parent))
            if not parent_path.endswith(".json"):
                parent_path += ".json"
            if parent_path in raw:
                merged.update(options(parent_path, depth + 1))
        merged.update(data.get("compilerOptions", {}) or {})
        return merged

    configs = []
    for path in raw:
        opts = options(path)
        directory = posixpath.dirname(path)
        configs.append(
            _TsConfig(
                directory=directory,
                base_url=opts.get("baseUrl"),
                paths={
                    k: list(v) for k, v in (opts.get("paths") or {}).items() if isinstance(v, list)
                },
            )
        )
    # Prefer the plain tsconfig.json in a folder over tsconfig.build.json etc.
    configs.sort(key=lambda c: (c.directory, 0 if c.paths else 1))
    return configs


class ImportResolver:
    """Resolves every file's imports within one service."""

    def __init__(self, files: dict[str, FileImports], sources: dict[str, str] | None = None):
        self.files = files
        self.paths = set(files)
        self.tsconfigs = _load_tsconfigs(sources or {})
        # JVM: fully qualified top-level name -> file declaring it.
        self.jvm_fqns: dict[str, str] = {}
        self.package_roots: set[str] = set()
        for info in files.values():
            if info.language in _JVM_LANGUAGES and info.package:
                # Three segments, not two: every service here lives under
                # `com.asurint`, so two would call an import of the shared
                # `com.asurint.models` library a missing local file.
                self.package_roots.add(".".join(info.package.split(".")[:3]))
                for name in info.top_level:
                    self.jvm_fqns.setdefault(f"{info.package}.{name}", info.path)
        self.unresolved: dict[str, list[str]] = {}
        self._cache: dict[tuple[str, str, str], tuple[str | None, bool]] = {}

    # --- JS / TS ---------------------------------------------------------

    def _tsconfig_for(self, path: str) -> list[_TsConfig]:
        directory = posixpath.dirname(path)
        found = [
            c
            for c in self.tsconfigs
            if directory == c.directory
            or directory.startswith(c.directory + "/")
            or c.directory == ""
        ]
        return sorted(found, key=lambda c: -len(c.directory))

    def _with_extensions(self, base: str) -> str | None:
        base = posixpath.normpath(base) if base else base
        if base in self.paths:
            return base
        for ext in _JS_EXTENSIONS:
            if base + ext in self.paths:
                return base + ext
        for index in _INDEX_FILES:
            candidate = posixpath.join(base, index)
            if candidate in self.paths:
                return candidate
        return None

    def _js_candidates(self, module: str, source: str) -> tuple[list[str], bool]:
        """(base paths to try, whether the module looked local)."""
        if module.startswith((".", "/")):
            return [posixpath.join(posixpath.dirname(source), module)], True
        bases: list[str] = []
        local = False
        for config in self._tsconfig_for(source):
            root = posixpath.normpath(posixpath.join(config.directory, config.base_url or "."))
            for pattern, targets in config.paths.items():
                if pattern.endswith("*"):
                    prefix = pattern[:-1]
                    if module.startswith(prefix):
                        local = True
                        rest = module[len(prefix) :]
                        bases += [posixpath.join(root, t.replace("*", rest)) for t in targets]
                elif module == pattern:
                    local = True
                    bases += [posixpath.join(root, t) for t in targets]
            if config.base_url is not None:
                bases.append(posixpath.join(root, module))
        return bases, local

    def resolve_module(self, module: str, source: str) -> tuple[str | None, bool]:
        """(the file `module` loads when imported from `source`, whether it
        looked local). Cached: following barrels resolves the same module
        from the same folder many times over."""
        info = self.files.get(source)
        language = info.language if info else ""
        key = (module, posixpath.dirname(source), language)
        if key not in self._cache:
            self._cache[key] = self._resolve_uncached(module, source, language)
        return self._cache[key]

    def _resolve_uncached(self, module: str, source: str, language: str) -> tuple[str | None, bool]:
        if language in _JS_LANGUAGES or language == "graphql":
            bases, local = self._js_candidates(module, source)
            for base in bases:
                found = self._with_extensions(base)
                if found:
                    return found, True
            return None, local
        if language in _JVM_LANGUAGES:
            return self._resolve_jvm(module)
        if language == "python":
            return self._resolve_python(module, source)
        return None, False

    def _defining_file(
        self, path: str, name: str, depth: int = 0, seen: frozenset = frozenset()
    ) -> tuple[str, list[str]]:
        """Follow re-exports from `path` to the file that defines `name`.
        Returns (file, barrels passed through)."""
        info = self.files.get(path)
        if info is None or depth > _MAX_BARREL_DEPTH or path in seen:
            return path, []
        if name in info.top_level:
            return path, []
        seen = seen | {path}
        for spec in info.imports:
            if not spec.reexport:
                continue
            target, _ = self.resolve_module(spec.module, path)
            if target is None:
                continue
            for original, alias in spec.names:
                if alias == name:
                    final, via = self._defining_file(target, original, depth + 1, seen)
                    return final, [path, *via]
            if spec.wildcard:
                final, via = self._defining_file(target, name, depth + 1, seen)
                if final != target or name in (
                    self.files.get(target).top_level if target in self.files else set()
                ):
                    return final, [path, *via]
        return path, []

    # --- JVM -------------------------------------------------------------

    def _resolve_jvm(self, module: str) -> tuple[str | None, bool]:
        parts = module.split(".")
        # `a.b.Outer.Inner` and a static `a.b.Util.method` live in the file
        # of their outermost declared prefix.
        for end in range(len(parts), 1, -1):
            found = self.jvm_fqns.get(".".join(parts[:end]))
            if found:
                return found, True
        return None, ".".join(parts[:3]) in self.package_roots

    # --- Python ----------------------------------------------------------

    def _resolve_python(self, module: str, source: str) -> tuple[str | None, bool]:
        if module.startswith("."):
            dots = len(module) - len(module.lstrip("."))
            base = posixpath.dirname(source)
            for _ in range(dots - 1):
                base = posixpath.dirname(base)
            rest = module.lstrip(".").replace(".", "/")
            roots = [posixpath.join(base, rest) if rest else base]
            local = True
        else:
            rest = module.replace(".", "/")
            roots = [rest, posixpath.join("src", rest)]
            local = False
        for root in roots:
            for candidate in (root + ".py", posixpath.join(root, "__init__.py")):
                if candidate in self.paths:
                    return candidate, True
        return None, local

    # --- all files -------------------------------------------------------

    def resolve_all(self) -> list[ResolvedImport]:
        edges: dict[tuple[str, str], ResolvedImport] = {}
        for path, info in self.files.items():
            for spec in info.imports:
                if spec.wildcard and info.language in _JVM_LANGUAGES:
                    continue  # a package, not a file; its names resolve as references
                target, local = self.resolve_module(spec.module, path)
                if target is None:
                    if local:
                        self.unresolved.setdefault(path, []).append(spec.module)
                    continue
                if target == path:
                    continue
                groups: dict[str, tuple[list[str], list[str]]] = {}
                names = spec.names if spec.names else []
                if info.language in _JS_LANGUAGES and names:
                    for original, alias in names:
                        if original in ("*", "default"):
                            groups.setdefault(target, ([], []))[0].append(alias)
                            continue
                        final, via = self._defining_file(target, original)
                        bucket = groups.setdefault(final, ([], via))
                        bucket[0].append(alias)
                else:
                    groups[target] = ([alias for _, alias in names], [])
                for final, (local_names, via) in groups.items():
                    if final == path:
                        continue
                    edge = edges.setdefault(
                        (path, final), ResolvedImport(path, final, reexport=spec.reexport)
                    )
                    edge.names = sorted(set(edge.names) | set(local_names))
                    edge.via = sorted(set(edge.via) | set(v for v in via if v != final))
        return list(edges.values())

    def local_bindings(self, path: str) -> dict[str, tuple[str, str]]:
        """{local name: (file it comes from, name there)} for one file's
        imports — what references.py resolves identifiers through."""
        info = self.files.get(path)
        bindings: dict[str, tuple[str, str]] = {}
        if info is None:
            return bindings
        for spec in info.imports:
            if spec.reexport:
                continue
            target, _ = self.resolve_module(spec.module, path)
            if target is None:
                continue
            for original, alias in spec.names:
                if original == "*":
                    continue
                if info.language in _JS_LANGUAGES and original != "default":
                    final, _ = self._defining_file(target, original)
                    bindings[alias] = (final, original)
                elif info.language in _JS_LANGUAGES:
                    bindings[alias] = (target, alias)
                else:
                    bindings[alias] = (target, original)
        return bindings
