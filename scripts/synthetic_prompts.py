"""Prompt construction for synthetic Python source generation.

Defines the (domain x style x length) sampling space and a deterministic
prompt builder used by ``scripts.gen_synthetic``. Pure stdlib; safe to
import in tests without any third-party dependency.
"""

from __future__ import annotations

import random
from typing import Final


# ---------------------------------------------------------------------------
# Sampling space
# ---------------------------------------------------------------------------

DOMAINS: Final[list[str]] = [
    "web scraping",
    "data analysis",
    "machine learning",
    "algorithms and data structures",
    "games",
    "command-line tools",
    "web servers (Flask/FastAPI)",
    "file processing",
    "image processing",
    "networking and sockets",
    "databases (SQLAlchemy/sqlite3)",
    "scientific computing (numpy/scipy)",
    "automation scripts",
    "physics or game simulators",
    "parsers",
    "compilers and interpreters",
    "testing tools (pytest plugins)",
    "system utilities",
    "GUI applications (tkinter/PyQt)",
    "audio processing",
    "API clients",
    "cryptography (educational)",
    "statistical analysis",
    "geographic / mapping",
    "recommendation systems",
    "graph algorithms",
    "concurrency (asyncio/threading)",
    "design patterns",
    "domain-specific languages",
    "configuration management",
]

STYLES: Final[list[str]] = [
    "idiomatic, type-annotated, with docstrings",
    "object-oriented with multiple classes",
    "functional style with pure functions and composition",
    "async/await throughout",
    "minimal procedural script",
    "well-tested with embedded pytest tests",
    "with inline type annotations and dataclasses",
    "using pattern matching (3.10+)",
    "verbose with comments explaining algorithms",
    "concise idiomatic",
]

# (label, approximate target line count)
LENGTH_TARGETS: Final[list[tuple[str, int]]] = [
    ("short", 80),
    ("medium", 200),
    ("long", 500),
]


# ---------------------------------------------------------------------------
# Per-domain seed topics for diversity
# ---------------------------------------------------------------------------

_TOPICS: Final[dict[str, list[str]]] = {
    "web scraping": [
        "scrape Hacker News front page into JSON",
        "extract product prices from a static HTML mirror",
        "crawl an internal wiki and dump titles + links",
        "scrape a sitemap.xml and download referenced PDFs",
        "build a polite multi-page scraper with retry/backoff",
    ],
    "data analysis": [
        "compute rolling statistics on a CSV of stock prices",
        "summarize a sales dataset by region and product",
        "detect outliers in a sensor time-series",
        "join two CSVs and produce a pivot table",
        "generate a Markdown report from a Parquet file",
    ],
    "machine learning": [
        "implement linear regression from scratch with gradient descent",
        "train a k-NN classifier on toy 2D data",
        "implement k-means clustering with seeded centroids",
        "tiny MLP forward/backward pass in pure NumPy",
        "logistic regression with L2 regularization",
    ],
    "algorithms and data structures": [
        "implement a balanced BST (AVL tree)",
        "Dijkstra's shortest path on an adjacency list",
        "LRU cache via OrderedDict and dict + doubly-linked list",
        "implement a trie for autocomplete",
        "Boyer-Moore string search",
        "segment tree with range-sum + point-update",
    ],
    "games": [
        "tic-tac-toe with minimax",
        "snake in the terminal using curses",
        "Conway's Game of Life",
        "tetris with a simple text renderer",
        "chess endgame solver for K+R vs K",
        "2048 in the terminal",
    ],
    "command-line tools": [
        "a `cat` clone with line numbering",
        "a `grep`-like tool with --color",
        "a directory size summarizer like `du`",
        "a clipboard-history CLI",
        "a tiny `make`-style task runner driven by YAML",
    ],
    "web servers (Flask/FastAPI)": [
        "a paste-bin service with FastAPI",
        "a JSON CRUD API for a todo list",
        "an OAuth-style login skeleton",
        "a small URL shortener with sqlite backing",
        "a webhook receiver that signs and forwards events",
    ],
    "file processing": [
        "deduplicate files by SHA-256 across a directory tree",
        "convert a directory of Markdown files to plain text",
        "find and rename files matching a glob pattern",
        "split a large JSONL file into N shards",
        "stream-process a gzipped log file line by line",
    ],
    "image processing": [
        "convert a folder of PNGs to grayscale JPEGs",
        "resize all images to fit within 1024x1024",
        "detect edges with a Sobel filter (NumPy only)",
        "make an animated GIF from a sequence of PNGs",
        "auto-crop whitespace from scanned pages",
    ],
    "networking and sockets": [
        "a TCP echo server with selectors",
        "a tiny chat server with multiple clients",
        "a UDP heartbeat protocol",
        "a port scanner with concurrent connect attempts",
        "an HTTP/1.0 server in raw sockets",
    ],
    "databases (SQLAlchemy/sqlite3)": [
        "a contacts CRUD app on sqlite3",
        "a SQLAlchemy ORM model for a small blog",
        "a migration runner for sqlite that applies SQL files in order",
        "a query builder for a single table",
        "a transactional bank-account toy with optimistic locking",
    ],
    "scientific computing (numpy/scipy)": [
        "FFT-based convolution of two signals",
        "solve a small linear system with iterative refinement",
        "Monte Carlo estimate of pi",
        "RK4 integrator for an ODE",
        "Gaussian-elimination solver in pure NumPy",
    ],
    "automation scripts": [
        "back up a directory to a timestamped tar.gz",
        "rotate log files older than N days",
        "watch a directory and print new files",
        "sync two directories one-way with hashing",
        "send a daily summary email from a JSON log",
    ],
    "physics or game simulators": [
        "2D N-body gravity simulator",
        "double pendulum integrator",
        "spring-mass cloth simulator",
        "1D heat equation finite-difference solver",
        "lattice gas / cellular fluid toy",
    ],
    "parsers": [
        "JSON parser by recursive descent",
        "INI file parser with section inheritance",
        "S-expression parser",
        "CSV parser handling quoted fields and escapes",
        "small arithmetic expression parser (Pratt parser)",
    ],
    "compilers and interpreters": [
        "tree-walking interpreter for a tiny calculator language",
        "stack-based bytecode VM with 8 opcodes",
        "BF (Brainfuck) interpreter",
        "tiny Lisp with lambda + lexical scope",
        "register-allocation toy on linear scan",
    ],
    "testing tools (pytest plugins)": [
        "a pytest plugin that records slowest tests",
        "a fixture that spins up a tmp sqlite db",
        "a hypothesis-style strategy builder (no external deps)",
        "snapshot-testing helper writing to .snap files",
        "parametrized matrix runner for env vars",
    ],
    "system utilities": [
        "a lightweight `ps`-like process listing",
        "a memory-usage watcher that alerts on threshold",
        "a uptime/idle-time reporter",
        "a tiny `cron` that reads a TOML schedule",
        "a disk-usage tree explorer",
    ],
    "GUI applications (tkinter/PyQt)": [
        "tkinter notepad with find-and-replace",
        "tkinter pomodoro timer",
        "tkinter color picker",
        "tkinter image viewer with thumbnails",
        "tkinter calculator with keyboard bindings",
    ],
    "audio processing": [
        "WAV file resampler (pure Python + struct)",
        "compute spectrogram with NumPy and dump as PNG-like array",
        "tone generator writing a WAV",
        "silence trimmer for WAV files",
        "biquad low-pass filter applied to a WAV",
    ],
    "API clients": [
        "GitHub issues client with pagination",
        "OpenWeather-style client with caching",
        "REST client with retry and rate-limit handling",
        "small JSON-RPC client over HTTP",
        "a typed wrapper around a public REST API",
    ],
    "cryptography (educational)": [
        "Caesar cipher and frequency-analysis cracker",
        "Vigenere cipher with Kasiski examination",
        "toy RSA with small primes",
        "HMAC-SHA256 from a SHA-256 implementation",
        "Diffie-Hellman key exchange demo over sockets",
    ],
    "statistical analysis": [
        "bootstrap confidence intervals for a sample mean",
        "two-sample t-test from scratch",
        "chi-square goodness-of-fit",
        "linear regression with diagnostics (residual plots)",
        "kernel density estimate on 1D data",
    ],
    "geographic / mapping": [
        "haversine distance and nearest-neighbor over points",
        "GeoJSON reader/writer roundtrip",
        "simple map-tile downloader with caching",
        "polygon point-in-test using ray casting",
        "GPX track simplifier (Douglas-Peucker)",
    ],
    "recommendation systems": [
        "user-based collaborative filtering on a tiny ratings matrix",
        "item-item cosine similarity recommender",
        "popularity-weighted recommender with cold start",
        "matrix factorization via SGD",
        "content-based recommender on TF-IDF vectors",
    ],
    "graph algorithms": [
        "Tarjan's strongly-connected components",
        "Kruskal's MST with union-find",
        "topological sort with cycle detection",
        "max flow via Edmonds-Karp",
        "A* on a 2D grid",
    ],
    "concurrency (asyncio/threading)": [
        "asyncio crawler with a bounded semaphore",
        "thread-pool fan-out/fan-in pattern",
        "producer/consumer with asyncio.Queue",
        "rate-limited scheduler using token bucket",
        "asyncio TCP proxy",
    ],
    "design patterns": [
        "observer pattern with a typed event bus",
        "state machine for a vending machine",
        "visitor pattern over an AST",
        "command pattern with undo/redo",
        "builder pattern for HTTP requests",
    ],
    "domain-specific languages": [
        "a regex-like matcher with NFA construction",
        "a tiny templating engine with for/if",
        "a configuration DSL parsed into dataclasses",
        "a query DSL compiled to SQL strings",
        "a make-like dependency DSL",
    ],
    "configuration management": [
        "merge layered YAML configs with override rules",
        "render Jinja-style templates without Jinja",
        "diff two TOML files semantically",
        "validate a config dict against a schema",
        "convert env vars into a nested config dict",
    ],
}

# Generic fallback if a domain has no entry above.
_GENERIC_TOPICS: Final[list[str]] = [
    "implement a small useful utility",
    "build a cohesive single-file demo",
    "write a focused, self-contained module",
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def random_seed_topic(domain: str, rng: random.Random) -> str:
    """Pick a concrete project topic for ``domain`` using ``rng``.

    Falls back to a generic topic if the domain is unknown. Always returns
    a non-empty string.
    """
    topics = _TOPICS.get(domain) or _GENERIC_TOPICS
    if not topics:
        topics = _GENERIC_TOPICS
    return rng.choice(topics)


def build_prompt(
    domain: str,
    style: str,
    target_lines: int,
    seed_topic: str | None = None,
) -> str:
    """Construct the user-facing prompt requesting one Python file.

    The returned text is a single string suitable as the user message body
    when calling Claude. The system prompt is provided separately by
    ``gen_synthetic`` (kept here to allow tests to assert on the user
    message without coupling to system-prompt formatting).
    """
    if not domain:
        raise ValueError("domain must be non-empty")
    if not style:
        raise ValueError("style must be non-empty")
    if target_lines <= 0:
        raise ValueError("target_lines must be positive")

    topic = seed_topic if seed_topic else "a focused, useful utility"

    return (
        f"Write ONE complete, runnable Python file in the domain of "
        f"{domain}.\n"
        f"Concrete topic: {topic}.\n"
        f"Style: {style}.\n"
        f"Target length: about {target_lines} lines (give or take 30%).\n"
        f"\n"
        f"Hard requirements:\n"
        f"- Output ONLY the raw Python source. No markdown code fences. "
        f"No commentary before or after. No explanations.\n"
        f"- The file must be syntactically valid (parseable by ast.parse) "
        f"and self-contained: no missing imports, no '...' placeholders, "
        f"no TODOs.\n"
        f"- Prefer the standard library. If you must use a third-party "
        f"library, pick a common one (numpy, requests, fastapi, "
        f"sqlalchemy, etc.) and import it normally.\n"
        f"- Include a `if __name__ == \"__main__\":` block when it makes "
        f"sense (CLIs, demos, simulators).\n"
        f"- Use clear names. Keep cyclomatic complexity reasonable.\n"
        f"- Do not produce trivial boilerplate; the file should actually "
        f"do the thing implied by the topic.\n"
        f"\n"
        f"Begin output with the very first line of the Python file."
    )


SYSTEM_PROMPT: Final[str] = (
    "You are a senior Python engineer producing high-quality training "
    "data for a code language model. You write only Python source code, "
    "never prose, never markdown. Every file you produce is complete, "
    "runnable, and free of placeholders. You prioritize correctness, "
    "readability, and idiomatic style."
)


__all__ = [
    "DOMAINS",
    "STYLES",
    "LENGTH_TARGETS",
    "SYSTEM_PROMPT",
    "build_prompt",
    "random_seed_topic",
]
