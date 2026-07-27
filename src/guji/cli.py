"""``guji`` command-line entry point (typer)."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import charreport, stats
from .config import load_config
from .models import Manifest
from .parse import dictionary, manifest, narrative, poetry

app = typer.Typer(add_completion=False, help="中文古籍 RAG 检索系统 — corpus tooling")
console = Console()
log = logging.getLogger("guji")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    _setup_logging(verbose)
    # load DASHSCOPE_API_KEY etc. from .env at the repo root (gitignored)
    from dotenv import load_dotenv

    load_dotenv()


@app.command()
def fetch():
    """Clone (or update) the corpus into data/raw/chtxt."""
    cfg = load_config()
    dest = cfg.raw_path
    if (dest / ".git").is_dir():
        log.info("corpus present, pulling latest: %s", dest)
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning %s -> %s", cfg.corpus.repo_url, dest)
        subprocess.run(
            ["git", "clone", "--depth", "1", cfg.corpus.repo_url, str(dest)], check=True
        )
    log.info("done.")


@app.command(name="manifest")
def manifest_cmd():
    """Scan the corpus and write manifest.json."""
    cfg = load_config()
    if not cfg.raw_path.is_dir():
        raise typer.BadParameter(f"corpus not found at {cfg.raw_path}; run `guji fetch` first")
    m = manifest.build_manifest(cfg.raw_path)
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(
        m.model_dump_json(indent=2), encoding="utf-8"
    )

    total = len(m.books)
    empty = sum(b.is_empty for b in m.books)
    stub = sum(b.is_stub for b in m.books)
    non_empty = total - empty
    log.info(
        "manifest: %d books (%d non-empty, %d empty, %d stub) -> %s",
        total, non_empty, empty, stub, cfg.manifest_path,
    )
    log.info("non-empty book count = %d (README ticked = 56)", non_empty)


@app.command()
def normalize(check: bool = typer.Option(False, "--check", help="scan PUA/ext-CJK and write reports")):
    """Normalization utilities. Use --check to profile problem characters."""
    if not check:
        raise typer.BadParameter("only --check is implemented in Phase 1")
    cfg = load_config()
    log.info("scanning corpus for PUA / extension-CJK characters ...")
    report = charreport.scan(cfg.raw_path)
    charreport.write_report(report, cfg.char_report_path)
    added = charreport.merge_pua_map(report, cfg.pua_map_path)
    log.info(
        "PUA: %d distinct / %d occ; ext-CJK: %d distinct / %d occ",
        report["pua"]["distinct"], report["pua"]["total_occurrences"],
        report["ext_cjk"]["distinct"], report["ext_cjk"]["total_occurrences"],
    )
    log.info("wrote %s; pua_map.json +%d new placeholders", cfg.char_report_path, added)


@app.command()
def parse():
    """Parse the corpus into passages.jsonl + dict.sqlite."""
    cfg = load_config()
    m = Manifest.model_validate_json(cfg.manifest_path.read_text(encoding="utf-8"))
    pua = str(cfg.pua_map_path)
    lo = cfg.chunk.narrative_min_chars
    hi = cfg.chunk.narrative_max_chars
    ov = cfg.chunk.overlap_paragraphs

    books = [b for b in m.books if not (b.is_empty or b.is_stub)]
    passage_books = [b for b in books if b.form in ("narrative", "poetry")]
    dict_books = [b for b in books if b.form == "dictionary"]

    # --- passages.jsonl (narrative + poetry) ---
    cfg.passages_path.parent.mkdir(parents=True, exist_ok=True)
    n_passages = 0
    with cfg.passages_path.open("w", encoding="utf-8") as out, Progress(console=console) as prog:
        task = prog.add_task("passages", total=len(passage_books))
        for b in passage_books:
            mod = narrative if b.form == "narrative" else poetry
            for p in mod.parse_book(b, cfg.raw_path, pua, lo, hi, ov):
                out.write(p.model_dump_json() + "\n")
                n_passages += 1
            prog.advance(task)
    log.info("wrote %d passages -> %s", n_passages, cfg.passages_path)

    # --- dict.sqlite (dictionaries) ---
    conn = dictionary.create_db(cfg.dict_db_path)
    n_entries = 0
    with Progress(console=console) as prog:
        task = prog.add_task("dictionary", total=len(dict_books))
        for b in dict_books:
            n_entries += dictionary.insert_entries(
                conn, dictionary.parse_book(b, cfg.raw_path, pua)
            )
            prog.advance(task)
    conn.close()
    log.info("wrote %d dict entries -> %s", n_entries, cfg.dict_db_path)


@app.command(name="stats")
def stats_cmd():
    """Show chunk/entry distribution and the corpus char-budget check."""
    cfg = load_config()
    stats.render(cfg, console)


@app.command()
def verify(n: int = typer.Option(20, help="random passages to spot-check")):
    """Phase 1 acceptance: 56 books, char budget ±2%, no PUA residue, samples."""
    import random

    cfg = load_config()
    m = Manifest.model_validate_json(cfg.manifest_path.read_text(encoding="utf-8"))
    non_empty = sum(not b.is_empty for b in m.books)
    ok_books = non_empty == 56

    s = stats.render(cfg, console)
    ok_budget = abs(s["unexplained_pct"]) <= 2.0
    ok_pua = s["pua_residue"] == 0

    console.rule("random passage spot-check")
    passages = stats._load_passages(cfg.passages_path)
    for p in random.sample(passages, min(n, len(passages))):
        console.print(f"[cyan]{p['chunk_id']}[/]  《{p['title']}》{p['juan']}")
        console.print(f"  {p['text_raw'][:60]}…")

    console.rule("verdict")
    for label, ok in [
        (f"non-empty books == 56 (got {non_empty})", ok_books),
        (f"char budget reconciled, unexplained ≤2% (got {s['unexplained_pct']:.2f}%)", ok_budget),
        (f"no PUA residue (got {s['pua_residue']})", ok_pua),
    ]:
        console.print(f"[{'green' if ok else 'red'}]{'PASS' if ok else 'FAIL'}[/]  {label}")
    raise typer.Exit(0 if (ok_books and ok_budget and ok_pua) else 1)


@app.command()
def embed(
    limit: int = typer.Option(0, help="embed at most N new passages (0 = all)"),
    book: list[str] = typer.Option(None, "--book", help="restrict to book_id(s)"),
    batch: int = typer.Option(32, help="Ollama embedding batch size"),
):
    """Embed passages into the LanceDB vector store (resumable, with ETA)."""
    from .index import embed as emb

    cfg = load_config()
    book_ids = set(book) if book else None
    cols = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
    ]
    with Progress(*cols, console=console) as prog:
        task = prog.add_task("embedding", total=0)
        already, added = emb.embed_corpus(
            cfg,
            limit=limit or None,
            book_ids=book_ids,
            batch=batch,
            on_start=lambda n: prog.update(task, total=n),
            on_batch=lambda k: prog.advance(task, k),
        )
    log.info("embed: %d already present, %d newly embedded (model=%s)",
             already, added, cfg.embedding.model)


@app.command()
def index():
    """Build the ANN vector index + FTS5 character-bigram sparse index."""
    from .index import fts, vector

    cfg = load_config()
    db = vector.connect(cfg.lancedb_path)
    table = vector.open_table(db, cfg.embedding.dim)
    log.info("vector: %s", vector.create_ann_index(table))

    log.info("building FTS5 char-bigram index ...")
    n = fts.build(cfg.fts_db_path, cfg.passages_path)
    log.info("fts: indexed %d passages -> %s", n, cfg.fts_db_path)


@app.command()
def search(
    query: str = typer.Argument(..., help="natural-language query"),
    top_k: int = typer.Option(8, "--top-k", "-k"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    hyde: bool = typer.Option(None, "--hyde/--no-hyde", help="default from config"),
    book: str = typer.Option(None, "--book", help="filter by book title or id"),
    dynasty: str = typer.Option(None, "--dynasty"),
    category: str = typer.Option(None, "--category"),
    debug: bool = typer.Option(False, "--debug", help="show per-path recall + HyDE text"),
):
    """Hybrid search: dense(+HyDE) + sparse → RRF → rerank → threshold → dedup."""
    from .retrieve import hybrid

    cfg = load_config()
    res = hybrid.search(
        cfg, query, top_k=top_k, use_hyde=hyde, use_rerank=rerank,
        book=book, dynasty=dynasty, category=category,
    )

    if debug:
        console.rule("debug")
        if res.hyde_text:
            console.print(f"[magenta]HyDE:[/] {res.hyde_text[:160]}")
        console.print(f"[blue]dense top:[/] {', '.join(res.dense_top) or '—'}")
        console.print(f"[blue]sparse top:[/] {', '.join(res.sparse_top) or '—'}")
        console.rule("results")

    if not res.hits:
        reason = "分數低於閾值" if res.rejected_by_threshold else "無召回"
        console.print(f"[yellow]未檢索到相關記載[/] [dim]({reason})[/]")
        raise typer.Exit(0)

    for i, h in enumerate(res.hits, 1):
        m = h.meta
        cite = f"《{m.get('title','?')}》{m.get('juan','')}".rstrip()
        prov = f"dense#{h.dense_rank}" if h.dense_rank is not None else "—"
        prov += f" sparse#{h.sparse_rank}" if h.sparse_rank is not None else ""
        rr = f"rerank={h.rerank_score:.3f} " if h.rerank_score is not None else ""
        console.print(f"[bold cyan]{i}.[/] {cite}  [dim]({h.chunk_id}  {rr}rrf={h.rrf_score:.4f}  {prov})[/]")
        console.print(f"   [green]{m.get('text_raw','')[:80]}[/]")


def _render_answer(console: Console, text: str, qopen: str, qclose: str) -> None:
    """Print model prose in the default style; verbatim quotes as a green blockquote."""
    import re

    pattern = re.compile(re.escape(qopen) + r"(.*?)" + re.escape(qclose), re.S)
    pos = 0
    for m in pattern.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            console.print(before)
        quote = m.group(1).strip()
        console.print(f"  [green]▎{qopen}{quote}{qclose}[/]")
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        console.print(tail)


@app.command()
def ask(
    query: str = typer.Argument(..., help="natural-language question"),
    book: str = typer.Option(None, "--book", help="filter by book title or id"),
    dynasty: str = typer.Option(None, "--dynasty"),
    category: str = typer.Option(None, "--category"),
    hyde: bool = typer.Option(None, "--hyde/--no-hyde", help="default from config"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    validate: bool = typer.Option(
        None, "--validate/--no-validate",
        help="citation/quote/link validation + reject-and-retry; default from config. "
             "--no-validate returns the model's first answer as-is, ungrounded.",
    ),
    debug: bool = typer.Option(False, "--debug", help="show tool calls + retry attempts"),
):
    """Citation-grounded Q&A: retrieval + LLM function-calling, verbatim quotes only.

    Structured citations and quoted spans are validated in code against this turn's
    retrieved passages; an unverifiable answer is rejected and retried once (§4.2).
    """
    from . import generate

    cfg = load_config()
    res = generate.answer(
        cfg, query, book=book, dynasty=dynasty, category=category,
        use_hyde=hyde, use_rerank=rerank, validate_answer=validate,
    )

    if debug:
        console.rule("debug: HyDE")
        console.print(f"[magenta]HyDE text:[/] {res.hyde_text or '(disabled / empty)'}")

        console.rule("debug: retrieval (seeded search_passages result)")
        if not res.retrieved:
            console.print("[yellow](no hits — rejected_by_threshold or empty search)[/]")
        for i, r in enumerate(res.retrieved, 1):
            console.print(
                f"[cyan]{i}.[/] {r['chunk_id']}  《{r['title']}》{r['juan']}  "
                f"rerank={r['rerank_score']}"
            )
            console.print(f"   [green]{r['text_raw'][:120]}[/]")

        if res.messages:
            console.rule("debug: composer messages")
            for m in res.messages:
                role = m.get("role")
                if role == "system":
                    console.print(f"[bold]system:[/]\n{m['content']}")
                elif role == "user":
                    console.print(f"[bold]user:[/] {m['content']}")
                elif role == "assistant":
                    if m.get("content"):
                        console.print(f"[bold]assistant:[/] {m['content']}")
                    for tc in m.get("tool_calls") or []:
                        fn = tc["function"]
                        console.print(f"[bold]assistant tool_call:[/] {fn['name']}({fn['arguments']})")
                elif role == "tool":
                    console.print(f"[dim]tool_result[{m.get('tool_call_id')}]:[/] {m['content'][:2000]}")
                console.print()

        console.rule("debug: tool calls (beyond the seeded search)")
        if not res.tool_calls:
            console.print("[dim](none)[/]")
        for tc in res.tool_calls:
            console.print(f"[blue]tool:[/] {tc.name}({tc.arguments}) -> {len(tc.result)} rows")
            for r in tc.result[:5]:
                console.print(f"   {r}")

        console.rule("debug: attempts")
        for i, text in enumerate(res.answer_attempts, 1):
            console.print(f"[bold]attempt {i}:[/] {text or '(empty)'}")
            if i <= len(res.attempt_violations):
                bad_citations, bad_quotes = res.attempt_violations[i - 1]
                if not bad_citations and not bad_quotes:
                    console.print("[green]  -> passed validation[/]")
                else:
                    for title, juan, reason in bad_citations:
                        if reason == "hallucinated_link":
                            console.print(f"[red]  -> hallucinated link:[/] {juan}")
                        else:
                            console.print(f"[red]  -> bad citation:[/] 《{title}》{juan} ({reason})")
                    for q in bad_quotes:
                        console.print(f"[red]  -> bad quote (not verbatim in retrieved text_raw):[/] {q[:60]}")
        console.print(f"[dim]attempts: {res.attempts}[/]")
        console.rule("answer")

    if res.rejected_by_threshold or res.citation_failure:
        console.print(f"[yellow]{res.text}[/]")
        raise typer.Exit(0)

    _render_answer(console, res.text, cfg.generate.quote_open, cfg.generate.quote_close)


@app.command(name="eval")
def eval_cmd(
    label: str = typer.Option(..., "--label", "-l", help="report name, e.g. baseline / no_hyde"),
    top_k: int = typer.Option(20, help="Recall@K"),
    citation: bool = typer.Option(True, "--citation/--no-citation", help="also run generation for citation accuracy"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="cross-encoder rerank + threshold gate"),
    hyde: bool = typer.Option(None, "--hyde/--no-hyde", help="default from config"),
    limit: int = typer.Option(0, help="evaluate only the first N questions (0 = all 80)"),
):
    """Phase 5: Recall@K (retrieval) + citation accuracy (generation) over eval/questions.jsonl."""
    from . import eval as evalmod

    cfg = load_config()
    questions = evalmod.load_questions(cfg.eval_questions_path, limit=limit or None)

    by_cat_n: dict[str, int] = {}
    for q in questions:
        by_cat_n[q["category"]] = by_cat_n.get(q["category"], 0) + 1
    log.info("evaluating %d questions: %s", len(questions), by_cat_n)

    with Progress(console=console) as prog:
        task = prog.add_task("eval", total=len(questions))
        results = []
        for q in questions:
            qr = evalmod.eval_recall(cfg, q, top_k=top_k, use_rerank=rerank, use_hyde=hyde)
            if citation:
                evalmod.eval_citation(cfg, q, qr, use_rerank=rerank, use_hyde=hyde)
            results.append(qr)
            prog.advance(task)

    summary = evalmod.summarize(results)
    snapshot = {
        "top_k": top_k, "hyde_enabled": cfg.hyde.enabled if hyde is None else hyde, "rerank_enabled": rerank,
        "rerank_backend": cfg.rerank.backend if rerank else "none",
        "rerank_threshold": cfg.rerank.threshold, "dedup_enabled": cfg.dedup.enabled,
    }
    report_path = evalmod.write_report(cfg, label, results, summary, snapshot)

    table = Table(title=f"eval report: {label}")
    table.add_column("category")
    table.add_column("n", justify="right")
    table.add_column(f"recall@{top_k}", justify="right")
    table.add_column("citation acc.", justify="right")
    for cat, c in sorted(summary["by_category"].items()):
        cite = f"{c['citation_accuracy']:.0%}" if c["citation_accuracy"] is not None else "—"
        table.add_row(cat, str(c["n"]), f"{c['recall_at_k']:.0%}", cite)
    o = summary["overall"]
    ocite = f"{o['citation_accuracy']:.0%}" if o["citation_accuracy"] is not None else "—"
    table.add_row("[bold]TOTAL", f"[bold]{o['n']}", f"[bold]{o['recall_at_k']:.0%}", f"[bold]{ocite}")
    console.print(table)
    log.info("wrote %s", report_path)


@app.command(name="eval-compare")
def eval_compare(labels: list[str] = typer.Argument(..., help="report labels to compare, e.g. baseline no_hyde")):
    """A/B compare eval reports previously written by `guji eval --label ...`."""
    cfg = load_config()
    reports = []
    for label in labels:
        path = cfg.eval_report_dir / f"{label}.json"
        if not path.is_file():
            raise typer.BadParameter(f"no report at {path}; run `guji eval --label {label}` first")
        reports.append((label, json.loads(path.read_text(encoding="utf-8"))))

    cats = sorted({c for _, r in reports for c in r["summary"]["by_category"]})
    table = Table(title="eval A/B comparison")
    table.add_column("category")
    for label, _ in reports:
        table.add_column(f"{label}\nrecall", justify="right")
        table.add_column(f"{label}\ncitation", justify="right")
    for cat in [*cats, "TOTAL"]:
        row = [cat]
        for _, r in reports:
            c = r["summary"]["by_category"].get(cat) if cat != "TOTAL" else r["summary"]["overall"]
            recall = f"{c['recall_at_k']:.0%}" if c and c["recall_at_k"] is not None else "—"
            cite = f"{c['citation_accuracy']:.0%}" if c and c["citation_accuracy"] is not None else "—"
            row += [recall, cite]
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    app()
