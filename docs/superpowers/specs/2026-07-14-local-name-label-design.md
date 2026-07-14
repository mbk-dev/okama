# Design: expose `local_name` and unify asset labels (issue #93)

## Problem

The okama data API now returns a `local_name` field — the security's name in its
native language/script (Cyrillic for MOEX, 中文 for SHG/SHE/HK, etc.) — alongside
the Latin `name`. `Asset._get_symbol_data` reads only `name` and drops `local_name`.
Collections (`AssetList`, `Portfolio`, `EfficientFrontier`) therefore cannot use the
native name as a chart/table label. Additionally, the label-selection API is
inconsistent across classes (`ticker_names: bool`, `tickers: str|list`,
`tickers: bool`, and an always-both weights table), so a third label type cannot be
added uniformly.

This design surfaces `local_name` on `Asset`, propagates it through the collections
via a parallel `local_names` map, and consolidates label selection behind one
resolver — while keeping every existing parameter working unchanged.

## Decisions (agreed with user)

- **Full consolidation** of label selection (not the minimal alternative).
- **Legacy params kept working with NO deprecation warnings** (silent internal mapping).
- **`Asset.info`** (raw API payload) is stored, for forward-compat with future fields.
- **`Portfolio.table`**: append a `local name` column **only when at least one asset
  has a non-null `local_name`** (data-dependent column; absent for baskets with no
  native names, so existing 3-column behaviour is preserved for them).

## Out of scope

- `wealth_indexes`, `drawdowns` and other raw data properties: their columns must
  stay symbols (the issue itself forbids mutating underlying data). Users relabel at
  plot time via the public `names` / `local_names` maps.
- CJK font handling: documentation note only, no code.

## Design

### 1. `Asset` (`okama/asset.py`)

In `_get_symbol_data`:

```python
self.info: dict = x                                  # raw API payload (forward-compat)
...
self.name: str = x["name"]
self.local_name: str | None = x.get("local_name")    # .get(): nullable + older API
```

- `x.get("local_name")` — nullable: most markets have no native source yet, and
  older API deployments omit the key.
- Add `local_name` to `__repr__`.

### 2. Base class `ListMaker` (`okama/common/make_asset_list.py`)

In `_make_list`, in the **same loop** that builds `names` (so insertion order stays
identical to `self.symbols` — see Invariant below), build:

```python
local_names[asset_item.symbol] = asset_item.local_name or asset_item.name
```

Return it from `_make_list`; unpack to `self.local_names` in `__init__` alongside
`self.names`. The `or asset_item.name` fallback guarantees a non-empty label even in
mixed baskets where only some assets have a native name.

Add one resolver, the single source of label truth:

```python
def _asset_labels(self, mode: str = "ticker") -> list[str]:
    if mode == "ticker":
        return list(self.symbols)
    if mode == "name":
        return list(self.names.values())
    if mode == "local_name":
        return list(self.local_names.values())
    raise ValueError(f"Unknown label mode: {mode!r}. Use 'ticker', 'name' or 'local_name'.")
```

Plus a small legacy-bool normalizer used by the frontier compat shims:

```python
@staticmethod
def _labels_mode_from_bool(ticker_names: bool) -> str:
    return "ticker" if ticker_names else "name"
```

**Invariant (critical for correctness):** `_asset_labels(mode)[i]` corresponds to
`self.symbols[i]` for every mode. The frontier sites do
`dict(zip(asset_labels, weights))`; misalignment would silently attribute weights to
the wrong asset. The invariant holds because `names` and `local_names` are built in
the same loop as the symbols, and dicts preserve insertion order. It is locked by an
explicit test (see Testing).

### 3. `AssetList.plot_assets` / `EfficientFrontier.plot_pair_ef`

`plot_assets(tickers=...)` accepts `"tickers" | "names" | "local_names" | list[str]`.
Add the `"local_names"` branch, resolving via `self._asset_labels("local_name")`; the
existing `"tickers"` / `"names"` / list branches route through the resolver too.
`plot_pair_ef` merely forwards `tickers` to `plot_assets`, so it inherits the new
value; only its docstring needs the extra option. Backward compatible.

### 4. `AssetList.describe`

`describe(tickers: bool | str = True)`:

- `True` → `"ticker"` (unchanged; no rename), `False` → `"name"` (unchanged).
- `"tickers"` / `"names"` / `"local_names"` → select that mode explicitly.

Column rename uses `{symbol: label}` from the resolver. Backward compatible (bool
still works exactly as before).

### 5. `EfficientFrontier` (single + multi period)

There are **two** legacy surfaces; keep **both**:

- multi-period: `ticker_names` property + setter that validates `isinstance(bool)`.
- single-period: `labels_are_tickers` — a plain public attribute (get/set).

Introduce a **new, separately-named** selector so the bool validation stays intact:

- Store `self._labels_mode: str` (default `"ticker"`), initialised from the ctor's
  `ticker_names: bool` (unchanged ctor signature).
- New property `labels` (get/set) accepting `"ticker" | "name" | "local_name"`,
  reading/writing `self._labels_mode`.
- Legacy compat shims map to/from `_labels_mode`:
  - `ticker_names` getter → `self._labels_mode == "ticker"`; setter (bool-validated)
    → `self._labels_mode = _labels_mode_from_bool(value)`.
  - `labels_are_tickers` (single) → property backed by the same mapping.
- Every `asset_labels = self.symbols if self.ticker_names else list(self.names.values())`
  site → `asset_labels = self._asset_labels(self._labels_mode)`.
- single-period `get_assets_tickers()` → returns `self._asset_labels(self._labels_mode)`.

Net effect: `ef.labels = "local_name"` selects native names for all frontier point
dicts / `ef_points` columns / transition-map legends; `ticker_names=` and
`labels_are_tickers=` keep working (True↔ticker, False↔name).

### 6. `Portfolio.table` (`okama/portfolios/core.py`)

Append a `local name` column (values from `self.local_names`) **only when at least
one asset has a non-null native name**:

```python
if any(a.local_name for a in self.asset_obj_dict.values()):
    x.insert(1, "local name", list(self.local_names.values()))
```

Baskets with no native names keep the current 3-column shape; the docstring gains a
second example showing the 4-column form on a MOEX/CJK ticker.

## Testing (TDD)

- **Asset**: mock symbol-info with and without `local_name`; assert `Asset.local_name`
  (value and `None`), `Asset.info` (raw dict), and `local_name` in `__repr__`.
- **local_names fallback**: mixed basket (one asset with a native name, one without) →
  `self.local_names` has the native name for the first and the Latin `name` for the
  second.
- **Alignment invariant**: 2-asset list whose names sort opposite to the symbols;
  assert `_asset_labels("name")[i]` / `_asset_labels("local_name")[i]` line up with
  `symbols[i]`, and assert one frontier point dict maps the right weight to the right
  label.
- **Label modes**: for `plot_assets`, `describe`, and a frontier point dict, assert
  the labels emitted for each of `ticker` / `name` / `local_name`.
- **Backward compat**: `ticker_names=True/False` and `labels_are_tickers=True/False`
  round-trip to ticker/name on both frontier classes; the multi-period setter still
  rejects non-bool.
- Test infra: `tests/helpers/factories.py::FakeAsset` gains a `local_name`
  attribute (default `None`). `FakeCurrencyAsset` does **not** need it (currency is
  not in `asset_obj_dict`, so `_make_list` never reads its `local_name`).

## Docs & examples

- Document `Asset.local_name` / `Asset.info`, the `"local_names"` label option, and
  `EfficientFrontier.labels`, with a MOEX or Chinese-ticker example.
- Add a note that CJK labels need a CJK-capable matplotlib font (default fonts render
  tofu boxes).
- `/examples` notebooks use `ticker_names=`, `describe(tickers=`, `plot_assets(tickers=`,
  `plot_pair_ef(tickers=` and `.table`; all keep working. Confirm via nbmake on the
  affected notebooks (06, 07, 03) since plain pytest does not execute them.
