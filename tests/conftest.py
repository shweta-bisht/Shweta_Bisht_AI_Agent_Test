"""Shared pytest fixtures for CommerceAI unit and mocked integration tests."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from PIL import Image


def create_sample_jpeg_bytes(color=(120, 80, 50), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_workbook(tmp_path: Path) -> Path:
    """Create a realistic synthetic multi-sheet Excel workbook matching the assessment schema."""
    vendor_rows = [
        {
            "SKU": "SKU-101",
            "Masked SKU": "SKU-101",
            "Vendor1 Name": "Test Vendor",
            "Category": "Furniture",
            "Color": " Black ",
            "Material Used": "Wood, Faux Leather",
            "Assembly Required": "Yes",
            "Brand": "TestBrand",
            "Bullet 1": "One power reclining chair with cup holders",
            "Bullet 2": "  Solid wood frame for durability  ",
            "Bullet 3": "Solid wood frame for durability",  # deliberate duplicate
            "Bullet 4": None,
            "Image URL 1": "https://example.com/images/SKU-101.jpg",
            "Weight1": 110,
            "Length1": 40,
            "Width1": 36,
            "Height1": 41,
            "Dim1": "40x36x41",
            "Ship Weight1": 122,
            "Ship Length1": 42,
            "Ship Width1": 30,
            "Ship Height1": 32,
        },
        {
            "SKU": "SKU-102",
            "Masked SKU": "SKU-102",
            "Vendor1 Name": "Test Vendor",
            "Category": "Furniture",
            "Color": "Walnut",
            "Material Used": "Engineered Wood",
            "Assembly Required": "Yes",
            "Brand": "TestBrand",
            "Bullet 1": "One coffee table with lower storage shelf",
            "Bullet 2": "Compact rectangular silhouette",
            "Bullet 3": "",
            "Bullet 4": None,
            "Image URL 1": "https://example.com/images/SKU-102.jpg",
            "Weight1": 45,
            "Length1": 47,
            "Width1": 23.5,
            "Height1": 17,
            "Dim1": "47x23.5x17",
            "Ship Weight1": 52,
            "Ship Length1": 50,
            "Ship Width1": 26,
            "Ship Height1": 6,
        },
        {
            "SKU": "SKU-103",
            "Masked SKU": "SKU-103",
            "Vendor1 Name": "Test Vendor",
            "Category": "Furniture",
            "Color": None,  # missing optional attribute
            "Material Used": "Metal, Tempered Glass",
            "Assembly Required": "No",
            "Brand": "TestBrand",
            "Bullet 1": "One TV stand media cabinet with fireplace",
            "Bullet 2": "Includes remote controlled heater",
            "Bullet 3": None,
            "Bullet 4": None,
            "Image URL 1": "https://example.com/images/SKU-103.jpg",
            "Weight1": 140,
            "Length1": 70,
            "Width1": 16,
            "Height1": 33,
            "Dim1": "70x16x33",
            "Ship Weight1": 155,
            "Ship Length1": 74,
            "Ship Width1": 20,
            "Ship Height1": 10,
        },
    ]

    sales_rows = [
        {
            "orderDate": "2025-07-05",
            "orderId": 1001,
            "customerReferenceNumber": "REF-1",
            "invoiceNumber": 1001,
            "itemName": "SL-9001",
            "Masked SKU": "SL-9001",
            "quantity": 2,
            "price": 150.0,
            "amount": 300.0,
            "Customer Name": "Buyer A",
            "orderStatus": "Processed",
            "shippingCarrier": "FEDX",
            "Vendor": "WareHouse",
            "vendorType": "Alpha Supply",
        },
        {
            "orderDate": "2025-07-15",
            "orderId": 1002,
            "customerReferenceNumber": "REF-2",
            "invoiceNumber": 1002,
            "itemName": "SL-9002",
            "Masked SKU": "SL-9002",
            "quantity": 1,
            "price": 200.0,
            "amount": 200.0,
            "Customer Name": "Buyer B",
            "orderStatus": "Processed",
            "shippingCarrier": "Carrier Not Mapped",
            "Vendor": "Import",
            "vendorType": "Beta Supply",
        },
        {
            "orderDate": "2025-08-03",
            "orderId": 1003,
            "customerReferenceNumber": "REF-3",
            "invoiceNumber": 1003,
            "itemName": "SL-9001",
            "Masked SKU": "SL-9001",
            "quantity": 3,
            "price": 150.0,
            "amount": 450.0,
            "Customer Name": "Buyer C",
            "orderStatus": "Processed",
            "shippingCarrier": "UPSN",
            "Vendor": "WareHouse",
            "vendorType": "Alpha Supply",
        },
        {
            "orderDate": "2025-08-10",
            "orderId": 1004,
            "customerReferenceNumber": "REF-4",
            "invoiceNumber": 1004,
            "itemName": "SL-9003",
            "Masked SKU": "SL-9003",
            "quantity": 0,
            "price": 100.0,
            "amount": 0.0,
            "Customer Name": "Buyer D",
            "orderStatus": "Cancelled",
            "shippingCarrier": "FEDX",
            "Vendor": "Import",
            "vendorType": "Gamma Supply",
        },
        {
            "orderDate": "2025-08-15",
            "orderId": 1005,
            "customerReferenceNumber": "REF-5",
            "invoiceNumber": 1005,
            "itemName": "SL-9004",
            "Masked SKU": "SL-9004",
            "quantity": 1,
            "price": 120.0,
            "amount": 120.0,
            "Customer Name": "Buyer E",
            "orderStatus": "OJ Shipping",
            "shippingCarrier": "LTL",
            "Vendor": "DropShip",
            "vendorType": "Delta Supply",
        },
        {
            "orderDate": "2025-08-20",
            "orderId": 1006,
            "customerReferenceNumber": "REF-6",
            "invoiceNumber": 1006,
            "itemName": "SL-9005",
            "Masked SKU": "SL-9005",
            "quantity": 1,
            "price": 80.0,
            "amount": 80.0,
            "Customer Name": "Buyer F",
            "orderStatus": "Pending",
            "shippingCarrier": "FEDX",
            "Vendor": "Closeout",
            "vendorType": "Epsilon Supply",
        },
    ]

    wb_path = tmp_path / "sample_workbook.xlsx"
    with pd.ExcelWriter(wb_path, engine="openpyxl") as writer:
        pd.DataFrame(vendor_rows).to_excel(writer, sheet_name="vendor data", index=False)
        pd.DataFrame(sales_rows).to_excel(writer, sheet_name="sales data", index=False)
    return wb_path
