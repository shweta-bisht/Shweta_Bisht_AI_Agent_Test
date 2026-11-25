"""Vendor product helpers for product-type inference and diverse SKU selection."""

from __future__ import annotations

import re
from typing import Dict, List

from schemas.product_schema import NormalizedVendorProduct


def infer_product_type(product: NormalizedVendorProduct) -> str:
    """Derive a clean, factual product type phrase from the vendor bullets and category."""
    if product.bullets:
        first_bullet = product.bullets[0]
        # Take leading noun phrase before 'with', 'includes', or comma when concise
        short = re.split(r"\b(?:with|includes|featuring)\b|,", first_bullet, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        short = re.sub(r"^(?:One|Two|Three|Four|Five|A|An)\s+", "", short, flags=re.IGNORECASE).strip()
        if short:
            return short[0].upper() + short[1:]
    if product.category:
        return product.category
    return "Furniture Item"


def build_product_display_name(product: NormalizedVendorProduct) -> str:
    """Construct a factual display name strictly from brand, colour, and product type."""
    parts = [
        p
        for p in (
            product.brand,
            product.color,
            infer_product_type(product),
        )
        if p
    ]
    return " ".join(parts) if parts else product.sku


def select_representative_products(
    products: List[NormalizedVendorProduct], count: int = 5
) -> List[NormalizedVendorProduct]:
    """Select up to `count` diverse vendor products across inferred product types without hardcoding SKUs."""
    if count <= 0 or not products:
        return []
    if len(products) <= count:
        return list(products)

    # Group products by inferred product type preserving first-seen order
    buckets: Dict[str, List[NormalizedVendorProduct]] = {}
    for product in products:
        ptype = infer_product_type(product).lower()
        buckets.setdefault(ptype, []).append(product)

    selected: List[NormalizedVendorProduct] = []
    selected_skus = set()

    # Round-robin across product type buckets so all distinct furniture types are represented
    while len(selected) < count:
        added_in_round = False
        for ptype_list in buckets.values():
            while ptype_list and ptype_list[0].sku in selected_skus:
                ptype_list.pop(0)
            if ptype_list:
                chosen = ptype_list.pop(0)
                selected.append(chosen)
                selected_skus.add(chosen.sku)
                added_in_round = True
                if len(selected) >= count:
                    break
        if not added_in_round:
            break

    return selected
