# Calendar rebalancing and multi-period batching design

Continue the approved performance scope after conditional Rebalance. Preserve all earlier local changes and their public behavior.

Calendar wealth: replace per-period pandas containers with a NumPy loop over the exact original `resample()` groups, preserving within-period compounding, skipped-calendar weight carry, threshold evaluation only at period ends, event dates/types, optional asset paths and index metadata. Use the path only for normal complete finite float64 monthly histories and compatible positional weights. Keep the original pandas calendar branch as fallback for gaps, missing/nonstandard data, subclasses and other unsupported inputs. Zero thresholds retain original calendar truthiness semantics.

Multi-period Monte Carlo/grid: prepare the existing EF period grouping and within-period growth once per request. Apply independent weights in NumPy chunks, then calculate risk/CAGR and construct the compatible output. No public tuning API or persistent/invalidation cache. Bound periods-by-portfolios intermediates with an element limit and a maximum chunk count. Preserve labels, bounds, order, validation, no per-weight cache growth, supported rebalancing periods, short-history None CAGR and scalar exceptional/low-risk fallbacks.

Alternatives: reusing a dense all-portfolios matrix violates the memory goal; global data caching introduces invalidation concerns. Rewriting scalar optimization or adding GPU/JIT dependencies is outside this stage. Keep original scalar formulas as fallback and independent frozen-source tests.

Python >=3.11, existing NumPy/pandas/Poetry only. No dependency or machine-wide settings changes. No commits, push, merge, deployment or tracker mutation authorization.
