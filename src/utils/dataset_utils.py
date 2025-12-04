from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Iterator, List, Tuple

import feedparser
import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def compute_category_targets(cfg: Dict) -> List[Tuple[str, int]]:
    """
    Compute how many papers to fetch per category, based on weights.
    Returns: list of (category_id, target_count)
    """
    cats = cfg["dataset"]["categories"]
    max_papers = cfg["dataset"]["max_papers"]

    weights = [c["weight"] for c in cats]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Total category weight must be > 0")

    # First, compute proportional allocation
    raw_alloc = [max_papers * w / total_weight for w in weights]
    int_alloc = [max(int(x), 1) for x in raw_alloc]

    # Adjust to match max_papers exactly (or as close as possible)
    current_total = sum(int_alloc)

    if current_total > max_papers:
        # Scale down proportionally
        factor = max_papers / current_total
        int_alloc = [max(int(n * factor), 1) for n in int_alloc]
        current_total = sum(int_alloc)

    # If still off due to rounding, fix by adding/subtracting one
    while current_total < max_papers:
        # add one to the largest-weight categories first
        max_idx = max(range(len(int_alloc)), key=lambda i: cats[i]["weight"])
        int_alloc[max_idx] += 1
        current_total += 1

    while current_total > max_papers:
        # subtract one from the smallest-weight categories first (but keep >= 1)
        min_idx = min(
            range(len(int_alloc)),
            key=lambda i: (cats[i]["weight"], int_alloc[i]),
        )
        if int_alloc[min_idx] > 1:
            int_alloc[min_idx] -= 1
            current_total -= 1
        else:
            # can't shrink anymore without going to 0; break
            break

    targets = [(c["id"], n) for c, n in zip(cats, int_alloc)]
    return targets


def _entry_year(entry) -> int | None:
    """
    Extract publication year from a feed entry.
    """
    # feedparser gives published_parsed or updated_parsed as a struct_time
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return entry.published_parsed.tm_year
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return entry.updated_parsed.tm_year
    return None


def fetch_papers_for_category(
    cat_id: str,
    limit: int,
    recent_years: int,
    batch_size: int,
    sleep_seconds: float,
) -> Iterator[Dict]:
    """
    Stream papers from arXiv for a single category.

    - Fetches in batches (batch_size) via the arXiv API.
    - Filters by publication year: only last `recent_years`.
    - Yields one record (dict) at a time.
    """
    fetched = 0
    start = 0

    current_year = datetime.utcnow().year
    cutoff_year = current_year - recent_years + 1

    while fetched < limit:
        remaining = limit - fetched
        max_results = min(batch_size, remaining)

        params = {
            "search_query": f"cat:{cat_id}",
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        entries = feed.entries

        if not entries:
            # No more results for this category
            break

        for entry in entries:
            year = _entry_year(entry)
            if year is not None and year < cutoff_year:
                # Since results are sorted by date desc, once we hit older
                # than cutoff_year, we can stop for this category.
                return

            arxiv_id = entry.get("id", "")
            # Typical arxiv id form: 'http://arxiv.org/abs/2101.00001v1'
            if "/abs/" in arxiv_id:
                arxiv_id = arxiv_id.split("/abs/")[-1]

            title = entry.get("title", "").replace("\n", " ").strip()
            abstract = entry.get("summary", "").replace("\n", " ").strip()

            authors = []
            for a in entry.get("authors", []):
                name = getattr(a, "name", None)
                if name:
                    authors.append(name)
            authors_str = ", ".join(authors)

            categories = []
            for tag in entry.get("tags", []):
                term = getattr(tag, "term", None)
                if term:
                    categories.append(term)
            categories_str = ", ".join(categories) if categories else cat_id

            record = {
                "id": arxiv_id,
                "title": title,
                "authors": authors_str,
                "categories": categories_str,
                "abstract": abstract,
            }

            yield record
            fetched += 1
            if fetched >= limit:
                break

        # Prepare for next page
        start += max_results

        # Polite delay
        time.sleep(sleep_seconds)


def fetch_papers_weighted(cfg: Dict) -> Iterator[Dict]:
    """
    Orchestrate weighted fetching across all categories.

    Uses `compute_category_targets` to decide how many papers
    to fetch per category.
    """
    targets = compute_category_targets(cfg)
    recent_years = cfg["dataset"]["recent_years"]
    batch_size = cfg["dataset"]["batch_size"]
    sleep_seconds = cfg["dataset"]["request_sleep_seconds"]

    total_yielded = 0
    max_papers = cfg["dataset"]["max_papers"]

    for cat_id, target_n in targets:
        if total_yielded >= max_papers:
            break

        # Don't exceed global max_papers
        remaining_global = max_papers - total_yielded
        target_for_cat = min(target_n, remaining_global)

        for rec in fetch_papers_for_category(
            cat_id=cat_id,
            limit=target_for_cat,
            recent_years=recent_years,
            batch_size=batch_size,
            sleep_seconds=sleep_seconds,
        ):
            yield rec
            total_yielded += 1
            if total_yielded >= max_papers:
                break
