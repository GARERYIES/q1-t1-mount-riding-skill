# Third-Party Notices and Excluded Assets

## 启元 Q1 / T1

Q1 and T1 are products of 上纬启元机器人 / PrimeBOT. Product names, robot
appearance, trademarks, official models, SDKs, and documentation remain the
property of their respective rights holders.

Official references:

- Q1 product: https://www.primebot.com/productinfo4.html
- Q1 specifications: https://www.primebot.com/product_spec4.html
- T1 product: https://www.primebot.com/t1.html
- T1 specifications: https://www.primebot.com/product_spec1.html
- Official developer downloads: https://dev.primebot.com/download

The official download page provides Q1/T1 URDF, STL, MuJoCo XML, SDK, and
documentation packages. It does **not** state a redistribution license for
those files in the public page reviewed for this repository. They are therefore
linked rather than vendored.

## Excluded from this public repository

- Q1/T1 STL meshes and combined MuJoCo XML;
- vendor or vendor-derived base policy files;
- project actor/checkpoint files;
- reference motion and reset-state banks;
- raw native trajectories and contact-force logs;
- training data derived from the restricted simulation stack.

These assets are not covered by the repository's MIT license.

## Demo media

`assets/Q1T1_mount_demo_handload_cross17_17s.mp4`,
`assets/demo_preview.gif`, `assets/demo_poster.png`, and
`assets/light_dr_ppo16_rejected_nominal.mp4` are project-generated technical
demonstrations. They are provided for viewing this project's research result.
No license is granted to reuse the depicted robot designs, trademarks, or
third-party model assets. All rights in those underlying elements remain with
their respective owners.

## Open-source software

This project depends on open-source software including MuJoCo, PyTorch, NumPy,
Pillow, and imageio-ffmpeg. Each dependency is governed by its own license.
Consult the installed package metadata and upstream repositories for exact
license terms.
