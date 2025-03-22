# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:56:21 2025

@author: mozgo
"""

target = 1_000_000  # Cel oszczędności
annual_deposit = 0.01  # Roczna wpłata
interest_rate = 0.05  # Oprocentowanie (5%)

years = 0
balance = 0

while balance < target:
    balance += annual_deposit  # Dodanie rocznej wpłaty
    balance *= (1 + interest_rate)  # Kapitalizacja odsetek
    years += 1

print(f"Liczba lat potrzebnych do osiągnięcia {target}: {years}")