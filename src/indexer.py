"""Simple repository indexer and vector store using sentence-transformers + Annoy.

This module creates a minimal on-disk index under `.cache/index` containing
an Annoy index and a JSON metadata file mapping vector ids to file locations.
Embeddings are computed with `sentence-transformers` when available; if not,
indexing will still collect files but embeddings will be absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from annoy import AnnoyIndex


CACHE_DIR = Path('.cache') / 'index'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _file_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


def _id_for_path(path: Path) -> str:
    return hashlib.sha1(str(path).encode('utf-8')).hexdigest()


class Indexer:
    def __init__(self, model_name: str | None = None, dim: int = 384):
        self.dim = dim
        self.model = None
        if _HAS_ST and model_name:
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        self.ann = AnnoyIndex(self.dim, 'angular')
        self.meta_path = CACHE_DIR / 'meta.json'
        self.ann_path = CACHE_DIR / 'ann.ann'
        self.meta = {}

    def index_paths(self, paths: List[Path]):
        id_counter = 0
        if self.meta_path.exists():
            try:
                self.meta = json.loads(self.meta_path.read_text())
                id_counter = len(self.meta)
            except Exception:
                self.meta = {}

        for p in paths:
            text = _file_text(p)
            if not text.strip():
                continue
            vid = _id_for_path(p)
            if vid in self.meta:
                continue
            self.meta[vid] = {'path': str(p), 'length': len(text)}
            if self.model:
                emb = self.model.encode(text, show_progress_bar=False)
                self.ann.add_item(id_counter, emb.tolist() if hasattr(emb, 'tolist') else emb)
                id_counter += 1

        # save meta and ann
        self.meta_path.write_text(json.dumps(self.meta, indent=2))
        if self.model:
            self.ann.build(10)
            self.ann.save(str(self.ann_path))

    def query(self, text: str, k: int = 5):
        if not self.model:
            raise RuntimeError('No embedding model available')
        qv = self.model.encode(text)
        ids = self.ann.get_nns_by_vector(qv.tolist() if hasattr(qv, 'tolist') else qv, k)
        results = []
        meta_items = list(self.meta.items())
        for i in ids:
            if i < len(meta_items):
                vid, info = meta_items[i]
                results.append(info)
        return results


def discover_repo_files(root: Path, exts: List[str] | None = None):
    exts = exts or ['.py', '.md', '.txt', '.js', '.ts']
    return [p for p in root.rglob('*') if p.suffix.lower() in exts]


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='.')
    p.add_argument('--model', type=str, default='all-MiniLM-L6-v2')
    args = p.parse_args()
    root = Path(args.root)
    files = discover_repo_files(root)
    idx = Indexer(model_name=args.model)
    idx.index_paths(files)
    print('Indexed', len(files))
