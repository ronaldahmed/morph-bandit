#!/bin/bash

declare -A iso_ud

iso_ud["en"]="UD_English-EWT UD_English-GUM UD_English-LinES UD_English-ParTUT UD_English-PUD"
iso_ud["cs"]="UD_Czech-CAC UD_Czech-CLTT UD_Czech-FicTree UD_Czech-PDT UD_Czech-PUD"
iso_ud["es"]="UD_Spanish-AnCora"
iso_ud["tr"]="UD_Turkish-IMST UD_Turkish-PUD"
iso_ud["ar"]="UD_Arabic-PADT UD_Arabic-PUD"
iso_ud["ja"]="UD_Japanese-GSD UD_Japanese-Modern UD_Japanese-PUD"

iso_ud["de"]="UD_German-GSD"
iso_ud["mt"]="UD_Maltese-MUDT"
iso_ud["shk"]="UD_Shipibo_Konibo-UFAL"

# German PUD gives psor Num,Gen,Per error