# SVP Video Pipeline

## Overview
（M5 で執筆）

## Installation
（M5 で執筆）

## Usage
（M5 で執筆）

## Architecture
（M5 で執筆）

## Known Limitations
- 仕様書からの逸脱:
  - duration_seconds: ge=1, le=15 → ge=4, le=15 に補正（fal.ai 実仕様）
  - aspect_ratio: 4 値 → 7 値に拡張（fal.ai 実仕様）
  - resolution: SVP 非搭載、CLI --cheap で 480p / それ以外 720p
- 未検証事項:
  - gpt-image-2 における forbidden 効力（M3 で評価予定）
  - motion_layer の最適記法（M4 で評価予定）
  - C 群リスク（逆手・線状物体・軟体）の動画化

## Development
（M5 で執筆）
