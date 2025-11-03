# pages/3_Financial_News.py
import streamlit as st
import feedparser
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import re
from collections import OrderedDict

st.set_page_config(page_title="Financial News", page_icon="📰", layout="wide")

# =========================
# DEDUPLICATION HELPERS
# =========================
TRACKING_PREFIXES = ("utm_", "mc_", "gclid", "fbclid", "icid")
STOP = {"the","a","an","live","updated","breaking","analysis","opinion"}

def normalize_url(u: str) -> str:
    pu = urlparse(u)
    q = [(k, v) for k, v in parse_qsl(pu.query, keep_blank_values=True)
         if not k.lower().startswith(TRACKING_PREFIXES)]
    pu = pu._replace(netloc=pu.netloc.lower().replace("www.", ""),
                     query=urlencode(q, doseq=True),
                     fragment="")
    return urlunparse(pu)

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s$%-]")  # keep $, %, - for tickers/percents

def normalize_title(t: str) -> str:
    t = t.lower()
    t = t.replace("&", " and ")
    t = _punct.sub(" ", t)
    t = _ws.sub(" ", t).strip()
    return t

def title_signature(t_norm: str) -> str:
    toks = [w for w in t_norm.split() if len(w) > 3 and w not in STOP]
    return " ".join(sorted(set(toks)))

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def is_near_duplicate(t1_norm: str, t2_norm: str) -> bool:
    return jaccard(title_signature(t1_norm), title_signature(t2_norm)) >= 0.8

def host_path(u: str) -> str:
    pu = urlparse(u)
    return pu.netloc + pu.path

def materialize(item, url_sig, title_norm, tsig):
    return {
        "title": item["title"],
        "title_norm": title_norm,
        "title_sig": tsig,
        "url": url_sig,
        "url_key": url_sig,                # exact URL signature
        "host_path": host_path(url_sig),   # domain+path signature
        "published": item.get("published_parsed") or item.get("published") or 0,
        "source": urlparse(url_sig).netloc,
    }

AGG = {"news.google.com", "finance.yahoo.com"}  # aggregators (prefer originals)

def is_preferable(new, old):
    new_host = urlparse(normalize_url(new["link"])).netloc
    old_host = urlparse(old["url"]).netloc
    # Prefer original publisher over aggregator
    if (old_host in AGG) and (new_host not in AGG):
        return True
    if (new_host in AGG) and (old_host not in AGG):
        return False
    # Otherwise, keep the earlier (often “original”) one
    n_pub = new.get("published_parsed") or 9e18
    return n_pub < old.get("published", 9e18)

clusters: OrderedDict[str, dict] = OrderedDict()

def add_item(item):
    url_sig = normalize_url(item["link"])
    title_norm = normalize_title(item["title"])
    tsig = title_signature(title_norm)

    for cid, keep in list(clusters.items()):
        # 1) exact same normalized URL
        if keep["url_key"] == url_sig:
            return
        # 2) same host+path (query ignored)
        if keep["host_path"] == host_path(url_sig):
            return
        # 3) exact same token-set signature
        if keep["title_sig"] == tsig:
            return
        # 4) fuzzy near-duplicate by Jaccard
        if is_near_duplicate(title_norm, keep["title_norm"]):
            if is_preferable(item, keep):
                clusters[cid] = materialize(item, url_sig, title_norm, tsig)
            return

    cid = f"{len(clusters)+1:05d}"
    clusters[cid] = materialize(item, url_sig, title_norm, tsig)

# =========================
# FEEDS
# =========================
FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",      # WSJ
    "https://www.investing.com/rss/news.rss",              # Investing.com
    "https://www.ft.com/?format=rss",                      # Financial Times
    "https://seekingalpha.com/market_currents.xml",        # Seeking Alpha
    "https://www.marketwatch.com/rss/topstories",          # MarketWatch
]

# =========================
# FETCH + BUILD CLUSTERS
# =========================
for url in FEEDS:
    try:
        feed = feedparser.parse(url)
        for e in feed.entries:
            title = (e.get("title") or "").strip()
            link  = (e.get("link") or "").strip()
            if not title or not link:
                continue
            add_item({
                "title": title,
                "link": link,
                "published_parsed": e.get("published_parsed")
            })
    except Exception:
        continue

# =========================
# RENDER
# =========================
st.title("Financial News")
st.caption("De-duplicated headlines from multiple financial sources (one link per story).")

for item in clusters.values():
    st.markdown(
        f"- [{item['title']}]({item['url']})  \n  <sub>({item['source']})</sub>",
        unsafe_allow_html=True
    )
