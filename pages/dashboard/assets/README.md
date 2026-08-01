# 资源说明

本目录存放 AstrNa 控制台页面随插件本地提供的模型、纹理、解码器与字体
资源，页面运行时不请求任何外部网络资源。

- `models/computer.glb` — CRT 电脑模型（压缩网格）。
- `textures/dirt.jpg`、`textures/grunge.webp`、`textures/rgba_noise.png` — 环境肌理。
- `textures/astrna_boot_screen.png`、`textures/astrna_boot_screen_mobile.png`
  — AstrNa 自有开机画面中性底图（不含版本号）。版本号由页面运行时叠加：
  启动页与二维 CRT 使用 DOM 版本标签，三维 CRT 屏幕使用 CanvasTexture
  在底图上动态绘制，版本唯一来源是插件 `metadata.yaml`。
- `textures/astrna_badge.png` — AstrNa 自有机身铭牌。
- `draco/` — 网格解压运行时。
- `fonts/stix-*.woff2` — 界面衬线字体。

开机画面与机身铭牌为 AstrNa 原创设计。
