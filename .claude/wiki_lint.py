#!/usr/bin/env python3
'''Health-check the wiki under .claude/. The mechanical half of the lint operation.

Catches the failure mode that kills a derived knowledge base: a page keeps
asserting a number the code no longer has. Everything it cannot check
mechanically -- contradictions, stale prose, missing pages -- is listed in
CLAUDE.md as the judgement half of the same operation.

    python3 .claude/wiki_lint.py

Exits 1 if anything is stale, so it can gate a commit.
'''
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI = os.path.join(ROOT, '.claude')
DOCS = os.path.join(WIKI, 'docs')

PENDULUM = 'generate_inverted_pendulum_trajectories.py'

# (label, owning wiki page, source file, pattern capturing the live value, how the wiki writes it)
#
# The owning page matters: a corpus-wide search would pass while the page that
# actually explains the constant had drifted, because some other page still
# happened to quote the number.
FACTS = [
    ('DATA_ROOT', 'datasets.md', PENDULUM, r"^DATA_ROOT = '([^']+)'", lambda v: v),
    ('grid resolution', 'datasets.md', PENDULUM, r'^GRID_RESOLUTION = ([\d.]+)', lambda v: v),
    ('train split size', 'datasets.md', PENDULUM, r'^DEFAULT_NUM_TRAJS = (\d+)', lambda v: f'{int(v):,}'),
    ('lqr horizon', 'datasets.md', PENDULUM, r"^DEFAULT_HORIZON = \{'lqr': (\d+)", lambda v: v),
    ('rl horizon', 'datasets.md', PENDULUM, r"^DEFAULT_HORIZON = \{'lqr': \d+, 'rl': (\d+)", lambda v: v),
    ('se_tol default', 'datasets.md', PENDULUM, r"'--se_tol'.*default=([\d.]+)", lambda v: v),
    ('max_batches default', 'datasets.md', PENDULUM, r"'--max_batches'.*default=(\d+)", lambda v: v),
    ('U_SAT', 'glossary.md', PENDULUM, r'^U_SAT = ([\d.]+)', lambda v: v),
]


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def wiki_pages():
    return sorted(f for f in os.listdir(DOCS) if f.endswith('.md'))


def states(page_text, value):
    '''Whole-token match, so 0.04 does not satisfy a page that says 0.045.'''
    return re.search(rf'(?<![\w.]){re.escape(value)}(?![\w.])', page_text) is not None


def check_facts(problems):
    '''Every constant the wiki quotes must still be the constant the code has.'''
    for label, owner, source, pattern, render in FACTS:
        text = read(os.path.join(ROOT, source))
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            problems.append(f'{label}: no longer extractable from {source} '
                            f'(pattern changed?) -- update .claude/wiki_lint.py')
            continue
        value = render(match.group(1))
        page = os.path.join(DOCS, owner)
        if not os.path.exists(page):
            problems.append(f'{label}: owning page docs/{owner} is missing')
        elif not states(read(page), value):
            problems.append(f'{label}: {source} says {value!r}, docs/{owner} does not')


def check_index(problems):
    '''index.md is the catalog: every page listed, nothing listed that is gone.'''
    index = read(os.path.join(WIKI, 'INDEX.md'))
    for page in wiki_pages():
        if page not in index:
            problems.append(f'INDEX.md does not catalog docs/{page}')
    for listed in set(re.findall(r'docs/([\w-]+\.md)', index)):
        if not os.path.exists(os.path.join(DOCS, listed)):
            problems.append(f'INDEX.md lists docs/{listed}, which does not exist')


def check_orphans(problems):
    '''A page nothing links to will not be found when it matters.'''
    linkers = {os.path.join(WIKI, 'INDEX.md'), os.path.join(ROOT, 'CLAUDE.md')}
    linkers |= {os.path.join(DOCS, p) for p in wiki_pages()}
    for page in wiki_pages():
        inbound = [src for src in linkers
                   if not src.endswith(page) and page in read(src)]
        if not inbound:
            problems.append(f'docs/{page} is an orphan -- no inbound links')


def check_shape(problems):
    '''Each page carries a title and the "load when" line the index mirrors.

    Also the cheapest possible guard against a page having been truncated to
    nothing by a bad edit -- every other check here passes on an empty file.
    '''
    for page in wiki_pages():
        text = read(os.path.join(DOCS, page))
        if not re.match(r'^# \S', text):
            problems.append(f'docs/{page} does not open with a "# Title" heading')
        if not re.search(r'^Load when ', text, re.MULTILINE):
            problems.append(f'docs/{page} has no "Load when ..." line')
        if 'Related:' not in text:
            problems.append(f'docs/{page} has no Related line')


def check_local_leaks(problems):
    '''Committed pages must stay correct on someone else's machine.

    The only absolute path the wiki is allowed to name is DATA_ROOT, which the
    generators hardcode and every collaborator shares. Anything else under
    /common/ is one person's home or env and belongs in CLAUDE.local.md.
    '''
    allowed = re.search(r"^DATA_ROOT = '([^']+)'", read(os.path.join(ROOT, PENDULUM)),
                        re.MULTILINE).group(1)
    for page in wiki_pages() + ['../INDEX.md', '../../CLAUDE.md']:
        text = read(os.path.join(DOCS, page))
        for path in set(re.findall(r'/(?:common|home|Users)/[\w./-]+', text)):
            if not path.startswith(allowed):
                problems.append(f'{os.path.basename(page)} names {path!r} -- '
                                f'machine-specific paths belong in CLAUDE.local.md')


def check_log(problems):
    '''Entries must keep the parseable prefix, or grep/tail stops working.'''
    path = os.path.join(WIKI, 'log.md')
    if not os.path.exists(path):
        problems.append('log.md is missing')
        return
    entries = re.findall(r'^## \[(\d{4}-\d{2}-\d{2})\] (\w+) \| ', read(path), re.MULTILINE)
    if not entries:
        problems.append('log.md has no entries matching "## [YYYY-MM-DD] <op> | <subject>"')
        return
    dates = [d for d, _ in entries]
    if dates != sorted(dates, reverse=True):
        problems.append('log.md entries are not newest-first')


def main():
    problems = []
    check_facts(problems)
    check_index(problems)
    check_orphans(problems)
    check_shape(problems)
    check_local_leaks(problems)
    check_log(problems)

    if not problems:
        print(f'wiki ok: {len(wiki_pages())} pages, {len(FACTS)} facts verified against source, '
              'no machine-specific paths')
        return 0
    for problem in problems:
        print(f'stale: {problem}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
